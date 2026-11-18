#!/usr/bin/env python3
"""Recipe preflight and environment tests without Docker, downloads, or live nodes."""

import builtins
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("recipe_runner", ROOT / "run-recipe.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ModelPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, {"HOME": str(self.root)}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.hub = self.root / ".cache/huggingface/hub"
        self.recipe = self.root / "recipe.yaml"
        self.recipe.write_text(
            'recipe_version: "1"\nname: Test model\ncontainer: test-image\n'
            'model: org/model\ncommand: vllm serve org/model\n'
        )
        self.process = self.patch(
            runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
        )
        self.image = self.patch(runner, "check_image_exists", return_value=True)
        self.build = self.patch(runner, "build_image", return_value=True)
        self.download = self.patch(runner, "download_model", side_effect=self.cache_model)
        self.patch(builtins, "input", side_effect=AssertionError("Unexpected prompt"))
        self.patch(runner, "ENV_FILE", new=None)

    def patch(self, target, name, **kwargs):
        mocked = patch.object(target, name, **kwargs)
        result = mocked.start()
        self.addCleanup(mocked.stop)
        return result

    def cache_model(self, *_args, hub=None):
        snapshot = (hub or self.hub) / "models--org--model/snapshots/commit"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text("{}")
        return True

    def run_recipe(self, *flags, cluster=False):
        mode = ["-n", "192.0.2.1,192.0.2.2"] if cluster else ["--solo"]
        argv = ["run-recipe.py", str(self.recipe), "--config", "/dev/null", *mode, *flags]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            result = runner.main()
        return result, output.getvalue()

    def test_missing_model_stops_solo_and_cluster_before_container_checks(self):
        for cluster in (False, True):
            with self.subTest(cluster=cluster):
                result, output = self.run_recipe(cluster=cluster)
                self.assertEqual(result, 1)
                self.assertIn("Model 'org/model' not found", output)
                self.assertIn("Rerun the same command with --setup", output)
                self.image.assert_not_called()
                self.build.assert_not_called()
                self.download.assert_not_called()
                self.process.assert_not_called()

    def test_cached_model_launches_without_setup(self):
        self.cache_model()
        result, output = self.run_recipe()
        self.assertEqual(result, 0, output)
        self.assertEqual(self.process.call_args.args[0][0], str(runner.LAUNCH_SCRIPT))
        self.download.assert_not_called()

    def test_setup_and_force_download_fetch_model_before_launch(self):
        for flag in ("--setup", "--force-download"):
            with self.subTest(flag=flag):
                self.download.reset_mock()
                self.process.reset_mock()
                result, output = self.run_recipe(flag)
                self.assertEqual(result, 0, output)
                self.download.assert_called_once_with("org/model", None)
                self.process.assert_called_once()

    def test_dry_run_and_build_only_allow_missing_model(self):
        for flag in ("--dry-run", "--build-only"):
            with self.subTest(flag=flag):
                result, output = self.run_recipe(flag)
                self.assertEqual(result, 0, output)
                self.download.assert_not_called()
                self.process.assert_not_called()

    def test_download_only_does_not_launch(self):
        result, output = self.run_recipe("--download-only")
        self.assertEqual(result, 0, output)
        self.download.assert_called_once_with("org/model", None)
        self.process.assert_not_called()

    def test_recipe_without_model_metadata_can_launch(self):
        self.recipe.write_text(self.recipe.read_text().replace("model: org/model\n", ""))
        result, output = self.run_recipe()
        self.assertEqual(result, 0, output)
        self.process.assert_called_once()

    def test_image_pin_is_optional_and_only_nonempty_pin_is_forwarded(self):
        base = self.recipe.read_text()
        pin = "sha256:" + "1" * 64
        for field, expected in (
            ("", None), ("image_id:\n", None),
            ('image_id: ""\n', None), (f"image_id: {pin}\n", pin),
        ):
            for cluster in (False, True):
                for dry_run in (False, True):
                    with self.subTest(field=field, cluster=cluster, dry_run=dry_run):
                        self.recipe.write_text(base + field)
                        self.cache_model()
                        self.process.reset_mock()
                        flags = ["--dry-run"] if dry_run else []
                        result, output = self.run_recipe(*flags, cluster=cluster)
                        self.assertEqual(result, 0, output)
                        if dry_run:
                            line = next(line for line in output.splitlines() if line.startswith("   ./launch-cluster.sh"))
                            command = shlex.split(line.rstrip(" \\"))
                        else:
                            command = self.process.call_args.args[0]
                        if expected is None:
                            self.assertNotIn("--image-id", command)
                        else:
                            self.assertEqual(command[command.index("--image-id") + 1], expected)

    def test_invalid_nonempty_image_pin_is_rejected(self):
        self.recipe.write_text(self.recipe.read_text() + "image_id: invalid\n")
        with self.assertRaises(SystemExit) as result, contextlib.redirect_stdout(io.StringIO()):
            runner.load_recipe(self.recipe)
        self.assertEqual(result.exception.code, 1)

    def test_cache_without_snapshots_is_missing(self):
        snapshots = self.hub / "models--org--model/snapshots"
        snapshots.mkdir(parents=True)
        result, output = self.run_recipe()
        self.assertEqual(result, 1)
        self.assertIn("--setup", output)
        self.process.assert_not_called()

    def test_custom_cache_paths_and_precedence(self):
        cases = [
            ({"XDG_CACHE_HOME": str(self.root / "xdg")}, self.root / "xdg/huggingface/hub"),
            ({"XDG_CACHE_HOME": "ignored", "HF_HOME": "~/custom cache"}, self.root / "custom cache/hub"),
            ({"HF_HOME": "ignored", "HUGGINGFACE_HUB_CACHE": "~/legacy hub"}, self.root / "legacy hub"),
            ({"HUGGINGFACE_HUB_CACHE": "ignored", "HF_HUB_CACHE": "$HOME/new hub"}, self.root / "new hub"),
        ]
        for variables, hub in cases:
            with self.subTest(variables=variables), patch.dict(os.environ, variables):
                self.assertFalse(runner.check_model_exists("org/model"))
                self.cache_model(hub=hub)
                result, output = self.run_recipe()
                self.assertEqual(result, 0, output)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recipe = self.root / "recipe.yaml"
        self.defaults = {
            "VLLM_PLE_TABLE_MEMORY": "shared",
            "B12X_DISK_BACKEND": "mmap",
            "NUMERIC": 4,
            "JSON_VALUE": '{"strategy": "SPREAD", "label": "a=b"}',
            "EMPTY": "",
        }
        self.recipe.write_text(runner.yaml.safe_dump({
            "recipe_version": "1", "name": "Environment test", "container": "test-image",
            "env": self.defaults,
            "command": 'printf "%s\\n" "$VLLM_PLE_TABLE_MEMORY" "$B12X_DISK_BACKEND"',
        }))

    def run_recipe(self, *flags, mode=("--solo",)):
        argv = ["run-recipe.py", str(self.recipe), "--config", "/dev/null", *mode, *flags]
        output = io.StringIO()
        launched = {}

        def launch(command):
            launched["args"] = command
            launched["script"] = Path(command[command.index("--launch-script") + 1]).read_text()
            return subprocess.CompletedProcess(command, 0)

        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), \
                patch.object(runner, "ENV_FILE", None), \
                patch.object(runner, "check_image_exists", return_value=True), \
                patch.object(runner.subprocess, "run", side_effect=launch):
            result = runner.main()
        self.assertEqual(result, 0, output.getvalue())
        return output.getvalue(), launched

    @staticmethod
    def env_args(args):
        return [args[i + 1] for i, arg in enumerate(args) if arg == "-e"]

    def test_recipe_environment_reaches_launcher_in_solo_native_and_ray_modes(self):
        for mode in (("--solo",), ("-n", "192.0.2.1,192.0.2.2"),
                     ("-n", "192.0.2.1,192.0.2.2", "--ray")):
            with self.subTest(mode=mode):
                _, launched = self.run_recipe(mode=mode)
                actual = self.env_args(launched["args"])
                self.assertCountEqual(actual, [f"{key}={value}" for key, value in self.defaults.items()])
                self.assertNotIn("export ", launched["script"])

    def test_cli_overrides_are_effective_and_dry_run_matches_launch(self):
        flags = ("-e", "VLLM_PLE_TABLE_MEMORY=shared", "-e", "VLLM_PLE_TABLE_MEMORY=disk",
                 "-e", "B12X_DISK_BACKEND=io_uring", "-e", "CLI_ONLY=a=b c", "-e", "NUMERIC=")
        _, launched = self.run_recipe(*flags)
        actual = self.env_args(launched["args"])
        expected = {**self.defaults, "VLLM_PLE_TABLE_MEMORY": "disk",
                    "B12X_DISK_BACKEND": "io_uring", "CLI_ONLY": "a=b c", "NUMERIC": ""}
        self.assertCountEqual(actual, [f"{key}={value}" for key, value in expected.items()])
        # Execute only the harmless printf recipe using the environment sent to Docker.
        result = subprocess.run(["bash", "-c", launched["script"]],
                                env=dict(item.split("=", 1) for item in actual),
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, "disk\nio_uring\n")
        output, dry_launch = self.run_recipe("--dry-run", *flags)
        self.assertEqual(dry_launch, {})
        launch_line = next(line for line in output.splitlines() if line.startswith("   ./launch-cluster.sh"))
        self.assertEqual(self.env_args(shlex.split(launch_line.rstrip(" \\"))), actual)
        self.assertNotIn("export ", output)

    def test_default_env_precedence_and_dry_run_match_in_all_modes(self):
        # Interleave options deliberately: precedence is by layer, not position.
        flags = ("-e", "B12X_DISK_BACKEND=first", "--default-env", "B12X_DISK_BACKEND=default",
                 "--default-env", "VLLM_PLE_TABLE_MEMORY=orchestrator",
                 "--default-env", "DEFAULT_ONLY=first", "-e", "B12X_DISK_BACKEND=final",
                 "--default-env", "DEFAULT_ONLY=a b=c", "--default-env", "EMPTY_DEFAULT=",
                 "--default-env", "JSON_VALUE=ignored", "--default-env", "NUMERIC=ignored")
        expected = {**self.defaults, "B12X_DISK_BACKEND": "final",
                    "DEFAULT_ONLY": "a b=c", "EMPTY_DEFAULT": ""}
        for mode in (("--solo",), ("-n", "192.0.2.1,192.0.2.2"),
                     ("-n", "192.0.2.1,192.0.2.2", "--ray")):
            with self.subTest(mode=mode):
                _, launched = self.run_recipe(*flags, mode=mode)
                actual = self.env_args(launched["args"])
                self.assertCountEqual(actual, [f"{key}={value}" for key, value in expected.items()])
                output, dry_launch = self.run_recipe("--dry-run", *flags, mode=mode)
                self.assertEqual(dry_launch, {})
                launch_line = next(line for line in output.splitlines() if line.startswith("   ./launch-cluster.sh"))
                self.assertEqual(self.env_args(shlex.split(launch_line.rstrip(" \\"))), actual)

    def test_bare_cli_variable_replaces_recipe_and_default_assignments(self):
        _, launched = self.run_recipe("--default-env", "VLLM_PLE_TABLE_MEMORY=default",
                                      "--default-env", "DEFAULT_ONLY=default",
                                      "-e", "VLLM_PLE_TABLE_MEMORY", "-e", "DEFAULT_ONLY")
        actual = self.env_args(launched["args"])
        self.assertIn("VLLM_PLE_TABLE_MEMORY", actual)
        self.assertIn("DEFAULT_ONLY", actual)
        self.assertNotIn("VLLM_PLE_TABLE_MEMORY=shared", actual)
        self.assertNotIn("DEFAULT_ONLY=default", actual)

    def test_default_env_rejects_missing_assignment_or_key(self):
        for assignment in ("BARE", "=value"):
            with self.subTest(assignment=assignment), \
                    patch.object(sys, "argv", ["run-recipe.py", str(self.recipe), "--solo",
                                               "--default-env", assignment]), \
                    contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as result:
                    runner.main()
                self.assertEqual(result.exception.code, 2)
                self.assertIn("must be KEY=VALUE", error.getvalue())

    def test_missing_empty_and_null_recipe_environment_allow_cli_variables(self):
        recipe = runner.yaml.safe_load(self.recipe.read_text())
        for env in (None, {}, "missing"):
            with self.subTest(env=env):
                if env == "missing":
                    recipe.pop("env", None)
                else:
                    recipe["env"] = env
                self.recipe.write_text(runner.yaml.safe_dump(recipe))
                _, launched = self.run_recipe("-e", "CLI_ONLY=value")
                self.assertEqual(self.env_args(launched["args"]), ["CLI_ONLY=value"])


class StagedWeightsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recipe = self.root / "recipe.yaml"
        self.write_recipe()

    def write_recipe(self, *, env=None, model="org/model", mods=None):
        self.recipe.write_text(runner.yaml.safe_dump({
            "recipe_version": "1", "name": "Staged test", "container": "test-image",
            "model": model, "mods": mods or [], "env": env or {},
            "command": "vllm serve org/model",
        }))

    def run_recipe(self, *flags, model_cached=True):
        argv = ["run-recipe.py", str(self.recipe), "--config", "/dev/null", "--solo", *flags]
        output = io.StringIO()
        launched = {}
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), \
                patch.object(runner, "ENV_FILE", None), \
                patch.object(runner, "check_model_exists", return_value=model_cached) as cached, \
                patch.object(runner, "check_image_exists", return_value=True), \
                patch.object(runner, "download_model", return_value=True) as download, \
                patch.object(runner.subprocess, "run", side_effect=lambda command: (
                    launched.update(args=command) or subprocess.CompletedProcess(command, 0)
                )):
            result = runner.main()
        return result, output.getvalue(), launched, cached, download

    @staticmethod
    def args_for(args, flag):
        return [args[i + 1] for i, arg in enumerate(args) if arg == flag]

    def assert_staged_launch(self, args, slug, weights_root):
        self.assertEqual(self.args_for(args, "-v"),
                         [f"{Path(weights_root) / slug}:/model:ro"])
        env = dict(item.split("=", 1) for item in EnvironmentTests.env_args(args))
        self.assertEqual(env["FES_WEIGHTS_SLUG"], slug)
        self.assertEqual(env["FES_HUB_MODEL"], "org/model")
        self.assertEqual(env["FES_WEIGHTS_ENABLED"], "1")
        self.assertEqual(env["FES_WEIGHTS_DIR"], "/model")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(self.args_for(args, "--apply-mod").count(str(runner.SCRIPT_DIR / "mods/fes-weights")), 1)

    def test_staged_mod_is_first_when_absent_from_recipe(self):
        self.write_recipe(env={"FES_WEIGHTS_SLUG": "slug"}, mods=["mods/needs-hub-shim"])
        result, output, launched, _, _ = self.run_recipe("--apply-mod", "mods/other-mod")
        self.assertEqual(result, 0, output)
        mods = self.args_for(launched["args"], "--apply-mod")
        self.assertEqual(mods[0], str(runner.SCRIPT_DIR / "mods/fes-weights"))
        self.assertEqual(mods.count(str(runner.SCRIPT_DIR / "mods/fes-weights")), 1)
        self.assertEqual(mods[1:], [
            str(runner.SCRIPT_DIR / "mods/needs-hub-shim"),
            str(Path.cwd() / "mods/other-mod"),
        ])

    def test_disabled_or_bare_enabled_value_fails_closed(self):
        for source in ("recipe", "default", "explicit"):
            for value in ("", "0", "false", None):
                with self.subTest(source=source, value=value):
                    assignment = "FES_WEIGHTS_ENABLED" if value is None else f"FES_WEIGHTS_ENABLED={value}"
                    if source == "default" and value is None:
                        continue
                    self.write_recipe(env={"FES_WEIGHTS_SLUG": "slug"})
                    flags = []
                    if source == "recipe":
                        self.write_recipe(env={"FES_WEIGHTS_SLUG": "slug", "FES_WEIGHTS_ENABLED": "" if value is None else value})
                    elif source == "default":
                        flags = ["--default-env", assignment]
                    else:
                        flags = ["-e", assignment]
                    result, output, launched, _, download = self.run_recipe(*flags)
                    self.assertEqual(result, 1)
                    self.assertIn("require FES_WEIGHTS_ENABLED=1", output)
                    self.assertEqual(launched, {})
                    download.assert_not_called()

    def test_recipe_altered_weights_dir_without_exact_mount_fails_closed(self):
        self.write_recipe(env={"FES_WEIGHTS_SLUG": "slug", "FES_WEIGHTS_DIR": "/elsewhere"})
        result, output, launched, _, download = self.run_recipe()
        self.assertEqual(result, 1)
        self.assertIn("requires an explicit read-only mount", output)
        self.assertEqual(launched, {})
        download.assert_not_called()

    def test_custom_weights_dir_requires_selected_readonly_mount(self):
        self.write_recipe(env={"FES_WEIGHTS_SLUG": "slug", "FES_WEIGHTS_DIR": "/custom"})
        result, output, launched, _, _ = self.run_recipe("-v", "/opt/llm/models/slug:/custom:ro")
        self.assertEqual(result, 0, output)
        self.assertEqual(self.args_for(launched["args"], "-v"), ["/opt/llm/models/slug:/custom:ro"])
        env = dict(item.split("=", 1) for item in EnvironmentTests.env_args(launched["args"]))
        self.assertEqual(env["FES_WEIGHTS_DIR"], "/custom")

    def test_explicit_custom_weights_dir_gets_matching_readonly_mount(self):
        self.write_recipe(env={"FES_WEIGHTS_SLUG": "slug"})
        result, output, launched, _, _ = self.run_recipe("-e", "FES_WEIGHTS_DIR=/custom")
        self.assertEqual(result, 0, output)
        self.assertEqual(self.args_for(launched["args"], "-v"), ["/opt/llm/models/slug:/custom:ro"])

    def test_recipe_slug_selects_mount_mod_and_offline_env(self):
        self.write_recipe(env={"FES_WEIGHTS_SLUG": "recipe.slug"}, mods=["mods/fes-weights"])
        root = self.root / "weights space"
        result, output, launched, cached, download = self.run_recipe(
            "--weights-root", str(root), "--apply-mod", "mods/fes-weights")
        self.assertEqual(result, 0, output)
        self.assert_staged_launch(launched["args"], "recipe.slug", root)
        cached.assert_not_called()
        download.assert_not_called()
        self.assertEqual(self.args_for(launched["args"], "--apply-mod").count(str(runner.SCRIPT_DIR / "mods/fes-weights")), 1)

    def test_slug_precedence_and_dry_run_quoting(self):
        self.write_recipe(env={"FES_WEIGHTS_SLUG": "recipe"})
        root = self.root / "root with spaces"
        result, output, launched, _, _ = self.run_recipe(
            "--default-env", "FES_WEIGHTS_SLUG=default", "--slug", "explicit", "--weights-root", str(root))
        self.assertEqual(result, 0, output)
        self.assert_staged_launch(launched["args"], "explicit", root)
        result, output, dry, _, _ = self.run_recipe(
            "--default-env", "FES_WEIGHTS_SLUG=default", "--slug", "explicit", "--weights-root", str(root), "--dry-run")
        self.assertEqual(result, 0, output)
        self.assertEqual(dry, {})
        line = next(line for line in output.splitlines() if line.startswith("   ./launch-cluster.sh"))
        self.assert_staged_launch(shlex.split(line.rstrip(" \\")), "explicit", root)

    def test_recipe_then_default_slug_precedence(self):
        self.write_recipe(env={"FES_WEIGHTS_SLUG": "recipe"})
        result, _, launched, _, _ = self.run_recipe("--default-env", "FES_WEIGHTS_SLUG=default")
        self.assertEqual(result, 0)
        self.assert_staged_launch(launched["args"], "recipe", "/opt/llm/models")
        self.write_recipe()
        result, _, launched, _, _ = self.run_recipe("--default-env", "FES_WEIGHTS_SLUG=default")
        self.assertEqual(result, 0)
        self.assert_staged_launch(launched["args"], "default", "/opt/llm/models")

    def test_no_fes_weights_forces_hub_mode_even_when_recipe_lists_mod(self):
        self.write_recipe(env={"FES_WEIGHTS_SLUG": "recipe", "HF_HUB_OFFLINE": "0"}, mods=["mods/fes-weights"])
        result, output, launched, cached, download = self.run_recipe(
            "--no-fes-weights", "--default-env", "FES_WEIGHTS_SLUG=default", "--setup")
        self.assertEqual(result, 0, output)
        self.assertEqual(self.args_for(launched["args"], "-v"), [])
        self.assertEqual(self.args_for(launched["args"], "--apply-mod"), [str(runner.SCRIPT_DIR / "mods/fes-weights")])
        env = dict(item.split("=", 1) for item in EnvironmentTests.env_args(launched["args"]))
        self.assertEqual(env["FES_WEIGHTS_SLUG"], "recipe")  # inert without ENABLED
        self.assertNotIn("FES_WEIGHTS_ENABLED", env)
        self.assertNotIn("FES_HUB_MODEL", env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "0")
        cached.assert_called_with("org/model")
        self.assertEqual(cached.call_count, 2)  # setup phase, then launch preflight
        download.assert_not_called()

    def test_malformed_or_empty_slug_fails_before_launch(self):
        for slug in ("", ".", "..", "../model", "nested/model", "/absolute", "-leading", "white space", "bad;value"):
            with self.subTest(slug=slug):
                self.write_recipe(env={"FES_WEIGHTS_SLUG": slug})
                result, output, launched, _, download = self.run_recipe()
                self.assertEqual(result, 1)
                self.assertIn("Invalid FES_WEIGHTS_SLUG", output)
                self.assertEqual(launched, {})
                download.assert_not_called()
        for slug in ("../model", "nested/model"):
            with self.subTest(cli=slug), patch.object(sys, "argv", [
                "run-recipe.py", str(self.recipe), "--solo", "--slug", slug
            ]), contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as result:
                    runner.main()
                self.assertEqual(result.exception.code, 2)
                self.assertIn("no slashes or traversal", error.getvalue())

    def test_empty_weights_root_fails_before_launch(self):
        self.write_recipe(env={"FES_WEIGHTS_SLUG": "slug"})
        result, output, launched, _, download = self.run_recipe("--weights-root", "")
        self.assertEqual(result, 1)
        self.assertIn("--weights-root cannot be empty", output)
        self.assertEqual(launched, {})
        download.assert_not_called()

    def test_selected_slug_replaces_conflicting_explicit_container_value(self):
        self.write_recipe(env={"FES_WEIGHTS_SLUG": "recipe"})
        result, output, launched, _, _ = self.run_recipe("-e", "FES_WEIGHTS_SLUG=")
        self.assertEqual(result, 0, output)
        self.assert_staged_launch(launched["args"], "recipe", "/opt/llm/models")

    def test_missing_hub_model_fails_closed_and_explicit_hub_model_recovers(self):
        self.write_recipe(model=None, env={"FES_WEIGHTS_SLUG": "slug"})
        result, output, launched, _, download = self.run_recipe()
        self.assertEqual(result, 1)
        self.assertIn("require a non-empty FES_HUB_MODEL", output)
        self.assertEqual(launched, {})
        download.assert_not_called()
        result, output, launched, _, _ = self.run_recipe("-e", "FES_HUB_MODEL=explicit/model")
        self.assertEqual(result, 0, output)
        env = dict(item.split("=", 1) for item in EnvironmentTests.env_args(launched["args"]))
        self.assertEqual(env["FES_HUB_MODEL"], "explicit/model")

    def test_staged_download_only_skips_hub_fetch_and_launch(self):
        self.write_recipe(env={"FES_WEIGHTS_SLUG": "slug"})
        result, output, launched, cached, download = self.run_recipe("--download-only")
        self.assertEqual(result, 0, output)
        self.assertEqual(launched, {})
        cached.assert_not_called()
        download.assert_not_called()

    def test_no_slug_preserves_legacy_hub_preflight(self):
        self.write_recipe(mods=["mods/fes-weights"])
        result, output, launched, _, download = self.run_recipe(model_cached=False)
        self.assertEqual(result, 1)
        self.assertEqual(launched, {})
        self.assertIn("not found in the local Hugging Face cache", output)
        download.assert_not_called()


class StagedDraftTests(unittest.TestCase):
    """Recipe-owned drafter mounts for external-draft spec-decode arms."""

    TARGET = "target"
    DRAFT = "draft-model"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.weights_root = self.root / "weights"
        self.recipe = self.root / "recipe.yaml"
        self.stage(self.TARGET)
        self.stage(self.DRAFT)
        self.write_recipe()

    def stage(self, slug, *, config=True):
        """A staged slug is a directory holding at least a config.json."""
        directory = self.weights_root / slug
        directory.mkdir(parents=True, exist_ok=True)
        if config:
            (directory / "config.json").write_text("{}")
        return directory

    def write_recipe(self, *, env=None, model="org/model", mods=None,
                     draft_slug=DRAFT, draft_hub_model="org/draft-model",
                     draft_dir=None, extra=None):
        recipe = {
            "recipe_version": "1", "name": "Draft test", "container": "test-image",
            "model": model, "mods": mods or [],
            "env": {"FES_WEIGHTS_SLUG": self.TARGET} if env is None else env,
            "command": "vllm serve org/model",
        }
        if draft_slug is not None:
            recipe["draft_slug"] = draft_slug
        if draft_hub_model is not None:
            recipe["draft_hub_model"] = draft_hub_model
        if draft_dir is not None:
            recipe["draft_dir"] = draft_dir
        recipe.update(extra or {})
        self.recipe.write_text(runner.yaml.safe_dump(recipe))

    def run_recipe(self, *flags):
        argv = ["run-recipe.py", str(self.recipe), "--config", "/dev/null", "--solo",
                "--weights-root", str(self.weights_root), *flags]
        output = io.StringIO()
        launched = {}
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), \
                patch.object(runner, "ENV_FILE", None), \
                patch.object(runner, "check_model_exists", return_value=True), \
                patch.object(runner, "check_image_exists", return_value=True), \
                patch.object(runner, "download_model", return_value=True) as download, \
                patch.object(runner.subprocess, "run", side_effect=lambda command: (
                    launched.update(args=command) or subprocess.CompletedProcess(command, 0)
                )):
            result = runner.main()
        return result, output.getvalue(), launched, download

    @staticmethod
    def launched_env(args):
        return dict(item.split("=", 1) for item in EnvironmentTests.env_args(args))

    def test_drafted_recipe_mounts_drafter_read_only_and_exports_env(self):
        result, output, launched, download = self.run_recipe()
        self.assertEqual(result, 0, output)
        args = launched["args"]
        # Exactly one drafter mount, beside the target mount, both read-only.
        self.assertEqual(StagedWeightsTests.args_for(args, "-v"), [
            f"{self.weights_root / self.DRAFT}:/drafter:ro",
            f"{self.weights_root / self.TARGET}:/model:ro",
        ])
        env = self.launched_env(args)
        self.assertEqual(env["FES_DRAFT_SLUG"], self.DRAFT)
        self.assertEqual(env["FES_DRAFT_DIR"], "/drafter")
        self.assertEqual(env["FES_DRAFT_ENABLED"], "1")
        self.assertEqual(env["FES_DRAFT_HUB_MODEL"], "org/draft-model")
        self.assertEqual(env["FES_WEIGHTS_SLUG"], self.TARGET)
        # One fes-weights mod verifies both mounts; nothing duplicates it.
        self.assertEqual(
            StagedWeightsTests.args_for(args, "--apply-mod").count(
                str(runner.SCRIPT_DIR / "mods/fes-weights")), 1)
        download.assert_not_called()

    def test_dry_run_renders_the_same_drafter_mount_and_env(self):
        result, output, launched, _ = self.run_recipe("--dry-run")
        self.assertEqual(result, 0, output)
        self.assertEqual(launched, {})
        line = next(line for line in output.splitlines()
                    if line.startswith("   ./launch-cluster.sh"))
        args = shlex.split(line.rstrip(" \\"))
        self.assertIn(f"{self.weights_root / self.DRAFT}:/drafter:ro",
                      StagedWeightsTests.args_for(args, "-v"))
        self.assertEqual(self.launched_env(args)["FES_DRAFT_DIR"], "/drafter")

    def test_recipe_without_drafter_mounts_nothing_and_exports_no_draft_env(self):
        self.write_recipe(draft_slug=None, draft_hub_model=None)
        result, output, launched, _ = self.run_recipe()
        self.assertEqual(result, 0, output)
        args = launched["args"]
        self.assertEqual(StagedWeightsTests.args_for(args, "-v"),
                         [f"{self.weights_root / self.TARGET}:/model:ro"])
        env = self.launched_env(args)
        self.assertEqual([key for key in env if key.startswith("FES_DRAFT_")], [])

    def test_draft_slug_cli_overrides_recipe_and_stays_single(self):
        self.stage("other-draft")
        self.write_recipe()
        result, output, launched, _ = self.run_recipe("--draft-slug", "other-draft")
        self.assertEqual(result, 0, output)
        args = launched["args"]
        self.assertEqual(StagedWeightsTests.args_for(args, "-v"), [
            f"{self.weights_root / 'other-draft'}:/drafter:ro",
            f"{self.weights_root / self.TARGET}:/model:ro",
        ])
        env = self.launched_env(args)
        self.assertEqual(env["FES_DRAFT_SLUG"], "other-draft")
        # The recipe's own draft_hub_model: stays the drafter's companion hub id.
        self.assertEqual(env["FES_DRAFT_HUB_MODEL"], "org/draft-model")

    def test_draft_hub_model_must_be_an_org_model_id(self):
        # draft_slug is a single path component, so the slug-derived default can
        # never satisfy the hub-shim shape mods/fes-weights/run.sh enforces after
        # docker run; the resolver rejects it on the host instead.
        for hub_model in (None, "no-slash", "org/a/b", "org/", "/model", "org /model",
                          "org/model name", "../org/model", "org/../model"):
            with self.subTest(draft_hub_model=hub_model):
                self.write_recipe(draft_hub_model=hub_model)
                result, output, launched, download = self.run_recipe()
                self.assertEqual(result, 1, output)
                self.assertIn("need an 'org/model' FES_DRAFT_HUB_MODEL", output)
                self.assertIn("draft_hub_model:", output)
                self.assertEqual(launched, {})
                download.assert_not_called()
        # A blank declaration is rejected even earlier, by the recipe reader.
        self.write_recipe(draft_hub_model="")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
                self.assertRaises(SystemExit) as result:
            runner.load_recipe(self.recipe)
        self.assertEqual(result.exception.code, 1)
        self.assertIn("'draft_hub_model' must be a non-empty", output.getvalue())
        # The two real NVIDIA drafter ids clear the shape gate.
        for hub_model in ("nvidia/Qwen3.5-4B-Qwen3.8-27B-Draft",
                          "nvidia/Qwen3.5-0.8B-Qwen3.8-27B-Draft"):
            with self.subTest(draft_hub_model=hub_model):
                self.write_recipe(draft_hub_model=hub_model)
                result, output, launched, _ = self.run_recipe()
                self.assertEqual(result, 0, output)
                self.assertEqual(
                    self.launched_env(launched["args"])["FES_DRAFT_HUB_MODEL"],
                    hub_model)
        # An explicit container-env value is the documented override path.
        self.write_recipe()
        result, output, launched, _ = self.run_recipe(
            "-e", "FES_DRAFT_HUB_MODEL=org/draft")
        self.assertEqual(result, 0, output)
        self.assertEqual(self.launched_env(launched["args"])["FES_DRAFT_HUB_MODEL"],
                         "org/draft")

    def test_no_fes_draft_ignores_the_recipe_drafter(self):
        result, output, launched, _ = self.run_recipe("--no-fes-draft")
        self.assertEqual(result, 0, output)
        args = launched["args"]
        self.assertEqual(StagedWeightsTests.args_for(args, "-v"),
                         [f"{self.weights_root / self.TARGET}:/model:ro"])
        env = self.launched_env(args)
        self.assertEqual([key for key in env if key.startswith("FES_DRAFT_")], [])

    def test_no_fes_weights_rejects_a_drafted_recipe(self):
        # mods/fes-weights/run.sh:21 exits early when FES_WEIGHTS_ENABLED is unset,
        # and --no-fes-weights makes the target resolver return before it writes that
        # key. Mounting a drafter in that mode would advertise /drafter while its
        # per-node verify and hub shim silently never ran, so it fails closed --
        # including for --dry-run, which is a flag-compatibility error, not a
        # missing-staged-weights error.
        for flags in (("--no-fes-weights",), ("--dry-run", "--no-fes-weights")):
            with self.subTest(flags=flags):
                result, output, launched, download = self.run_recipe(*flags)
                self.assertEqual(result, 1, output)
                self.assertIn("require staged-weights mode", output)
                self.assertIn("--no-fes-draft", output)
                self.assertEqual(launched, {})
                download.assert_not_called()
        # Both flags together is an explicit opt-out and stays in hub-cache mode.
        result, output, launched, _ = self.run_recipe(
            "--no-fes-weights", "--no-fes-draft")
        self.assertEqual(result, 0, output)
        args = launched["args"]
        self.assertEqual(StagedWeightsTests.args_for(args, "-v"), [])
        env = self.launched_env(args)
        self.assertEqual([key for key in env if key.startswith("FES_DRAFT_")], [])
        self.assertNotIn("FES_WEIGHTS_ENABLED", env)
        # A recipe without a drafter keeps the pre-existing hub-cache behaviour.
        self.write_recipe(draft_slug=None, draft_hub_model=None)
        result, output, launched, _ = self.run_recipe("--no-fes-weights")
        self.assertEqual(result, 0, output)
        self.assertEqual([key for key in self.launched_env(launched["args"])
                          if key.startswith("FES_DRAFT_")], [])

    def test_recipe_draft_dir_moves_the_mount_and_env_together(self):
        self.write_recipe(draft_dir="/draft")
        result, output, launched, _ = self.run_recipe()
        self.assertEqual(result, 0, output)
        args = launched["args"]
        self.assertIn(f"{self.weights_root / self.DRAFT}:/draft:ro",
                      StagedWeightsTests.args_for(args, "-v"))
        self.assertEqual(self.launched_env(args)["FES_DRAFT_DIR"], "/draft")

    def test_unstaged_or_incomplete_drafter_fails_before_launch(self):
        import shutil
        shutil.rmtree(self.weights_root / self.DRAFT)
        result, output, launched, download = self.run_recipe()
        self.assertEqual(result, 1)
        self.assertIn("Draft weights are not staged", output)
        self.assertEqual(launched, {})
        download.assert_not_called()
        self.stage(self.DRAFT, config=False)
        result, output, launched, download = self.run_recipe()
        self.assertEqual(result, 1)
        self.assertIn("contain no config.json", output)
        self.assertEqual(launched, {})
        download.assert_not_called()

    def test_preview_and_setup_phases_do_not_require_a_staged_drafter(self):
        import shutil
        shutil.rmtree(self.weights_root / self.DRAFT)
        for flag in ("--dry-run", "--build-only", "--download-only"):
            with self.subTest(flag=flag):
                result, output, launched, _ = self.run_recipe(flag)
                self.assertEqual(result, 0, output)
                if flag == "--dry-run":
                    # The preview still renders the mount a real launch would use.
                    line = next(line for line in output.splitlines()
                                if line.startswith("   ./launch-cluster.sh"))
                    self.assertIn(
                        f"{self.weights_root / self.DRAFT}:/drafter:ro",
                        StagedWeightsTests.args_for(shlex.split(line.rstrip(" \\")), "-v"))
                else:
                    self.assertEqual(launched, {})
        # The same unstaged drafter still fails a real launch.
        result, output, launched, _ = self.run_recipe()
        self.assertEqual(result, 1)
        self.assertIn("Draft weights are not staged", output)
        self.assertEqual(launched, {})

    def test_path_like_draft_slug_is_rejected(self):
        for slug in ("../etc", "a/b"):
            with self.subTest(recipe=slug):
                self.write_recipe(draft_slug=slug)
                output = io.StringIO()
                with contextlib.redirect_stdout(output), \
                        self.assertRaises(SystemExit) as exit_context:
                    runner.load_recipe(self.recipe)
                self.assertEqual(exit_context.exception.code, 1)
                self.assertIn("Recipe 'draft_slug' is invalid", output.getvalue())
                self.assertIn("no slashes or traversal", output.getvalue())
        self.write_recipe()
        for slug in ("../etc", "a/b"):
            with self.subTest(cli=slug), patch.object(sys, "argv", [
                "run-recipe.py", str(self.recipe), "--solo", "--draft-slug", slug
            ]), contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as exit_context:
                    runner.main()
                self.assertEqual(exit_context.exception.code, 2)
                self.assertIn("no slashes or traversal", error.getvalue())

    def test_invalid_draft_dir_fails_before_launch(self):
        for draft_dir in ("drafter", "", "/", "//", "../drafter", "/drafter/../..",
                          "/a/../b", "/a/../../etc", "/a/./b"):
            with self.subTest(draft_dir=draft_dir):
                self.write_recipe(draft_dir=draft_dir)
                output = io.StringIO()
                with contextlib.redirect_stdout(output), \
                        self.assertRaises(SystemExit) as exit_context:
                    runner.load_recipe(self.recipe)
                self.assertEqual(exit_context.exception.code, 1)
                self.assertIn("Recipe 'draft_dir' is invalid", output.getvalue())

    def test_unknown_recipe_field_fails_before_launch(self):
        self.write_recipe(extra={"draft_slugs": self.DRAFT})
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
                self.assertRaises(SystemExit) as exit_context:
            runner.load_recipe(self.recipe)
        self.assertEqual(exit_context.exception.code, 1)
        self.assertIn("unknown field(s): draft_slugs", output.getvalue())


class FesWeightsModTests(unittest.TestCase):
    def run_mod(self, *, enabled=True, valid=True, hub_model="org/model"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights"
            if valid:
                weights.mkdir()
                (weights / "config.json").write_text("{}")
                (weights / "model.safetensors").write_bytes(b"weights")
            env = dict(os.environ, HF_HOME=str(root / "cache"))
            if enabled:
                env.update(FES_WEIGHTS_ENABLED="1", FES_WEIGHTS_DIR=str(weights), FES_HUB_MODEL=hub_model)
            else:
                env.pop("FES_WEIGHTS_ENABLED", None)
            result = subprocess.run(["bash", str(ROOT / "mods/fes-weights/run.sh")],
                                    env=env, capture_output=True, text=True)
            shim = root / "cache/hub/models--org--model/snapshots/fes-staged/model.safetensors"
            return result, shim.read_bytes() if shim.exists() else None

    def test_enabled_mod_verifies_and_creates_hub_shim(self):
        result, shim = self.run_mod()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FES_WEIGHTS_OK op=verify", result.stdout)
        self.assertEqual(shim, b"weights")

    def test_missing_weights_fail_before_shim(self):
        result, shim = self.run_mod(valid=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FES_WEIGHTS_FAIL op=verify", result.stderr)
        self.assertIn("mount missing", result.stderr)
        self.assertIsNone(shim)

    def test_disabled_mod_remains_inert(self):
        result, shim = self.run_mod(enabled=False, valid=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FES_WEIGHTS_SKIP", result.stdout)
        self.assertIsNone(shim)


class FesEvalTests(unittest.TestCase):
    def test_wrapper_defaults_recipe_values_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            (root / "recipes").mkdir()
            (root / "mods/fes-weights").mkdir(parents=True)
            import shutil
            shutil.copy2(ROOT / "scripts/fes-eval.sh", root / "scripts/fes-eval.sh")
            shutil.copy2(ROOT / "run-recipe.py", root / "run-recipe.py")
            shutil.copy2(ROOT / "cluster_topology.py", root / "cluster_topology.py")
            (root / "mods/fes-weights/verify.py").write_text("import sys\nsys.exit(0)\n")
            (root / "mods/fes-weights/run.sh").write_text("#!/bin/bash\nexit 0\n")
            (root / "run-recipe.sh").write_text('#!/bin/bash\nexec python3 "$(dirname "$0")/run-recipe.py" "$@"\n')
            (root / "run-recipe.sh").chmod(0o755)
            recipe = {"recipe_version": "1", "name": "FES test", "container": "test",
                      "model": "org/model", "command": "vllm serve test",
                      "env": {"FES_WEIGHTS_ENABLED": "1", "FES_WEIGHTS_DIR": "/model",
                              "FES_HUB_MODEL": "recipe/model", "HF_HUB_OFFLINE": "0"}}
            recipe_path = root / "recipes/test.yaml"
            recipe_path.write_text(runner.yaml.safe_dump(recipe))
            base = ["bash", str(root / "scripts/fes-eval.sh"), "--weights-root", str(root / "weights space"),
                    "--hub-model", "wrapper/model", "test", "slug", "--solo", "--config", "/dev/null", "--dry-run"]
            for explicit in (False, True):
                flags = ["-e", "FES_WEIGHTS_ENABLED=1", "-e", "FES_WEIGHTS_DIR=/override path",
                         "-e", "FES_HUB_MODEL=override/model", "-e", "HF_HUB_OFFLINE=override"] if explicit else []
                result = subprocess.run([*base, *flags], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                line = next(line for line in result.stdout.splitlines() if line.startswith("   ./launch-cluster.sh"))
                actual = dict(item.split("=", 1) for item in EnvironmentTests.env_args(shlex.split(line.rstrip(" \\"))))
                expected = dict(item.split("=", 1) for item in flags[1::2]) if explicit else recipe["env"]
                for key, value in expected.items():
                    self.assertEqual(actual[key], value)
            recipe.pop("env")
            recipe_path.write_text(runner.yaml.safe_dump(recipe))
            result = subprocess.run(base, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            line = next(line for line in result.stdout.splitlines() if line.startswith("   ./launch-cluster.sh"))
            actual = dict(item.split("=", 1) for item in EnvironmentTests.env_args(shlex.split(line.rstrip(" \\"))))
            self.assertEqual(actual, {"FES_WEIGHTS_ENABLED": "1", "FES_WEIGHTS_DIR": "/model",
                                      "FES_WEIGHTS_SLUG": "slug", "FES_HUB_MODEL": "wrapper/model",
                                      "HF_HUB_OFFLINE": "1"})

    def test_wrapper_no_fes_weights_skips_staged_verification_and_mount(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            (root / "recipes").mkdir()
            (root / "mods/fes-weights").mkdir(parents=True)
            import shutil
            shutil.copy2(ROOT / "scripts/fes-eval.sh", root / "scripts/fes-eval.sh")
            shutil.copy2(ROOT / "run-recipe.py", root / "run-recipe.py")
            shutil.copy2(ROOT / "cluster_topology.py", root / "cluster_topology.py")
            (root / "mods/fes-weights/verify.py").write_text("raise AssertionError('must not verify in hub mode')\n")
            (root / "run-recipe.sh").write_text('#!/bin/bash\nexec python3 "$(dirname "$0")/run-recipe.py" "$@"\n')
            (root / "run-recipe.sh").chmod(0o755)
            (root / "recipes/test.yaml").write_text(runner.yaml.safe_dump({
                "recipe_version": "1", "name": "Hub test", "container": "test",
                "model": "org/model", "command": "vllm serve test",
                "env": {"FES_WEIGHTS_SLUG": "recipe", "HF_HUB_OFFLINE": "0"},
            }))
            result = subprocess.run([
                "bash", str(root / "scripts/fes-eval.sh"), "--no-fes-weights",
                "test", "not-used", "--solo", "--config", "/dev/null", "--dry-run",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("--no-fes-weights", result.stdout)
            self.assertNotIn("/model:ro", result.stdout)
            self.assertNotIn("FES_WEIGHTS_ENABLED", result.stdout)
            self.assertNotIn("FES_HUB_MODEL", result.stdout)
            self.assertNotIn("HF_HUB_OFFLINE=1", result.stdout)


class FesRecipeCatalogTests(unittest.TestCase):
    def test_unambiguous_cluster_mappings_are_literal_recipe_declarations(self):
        mappings = {
            "mimo-v2.6-flash-exl3-1x": "mimo-v2.6-flash-rl-exl3-v2",
            "mimo-v2.6-flash-exl3-2x": "mimo-v2.6-flash-rl-exl3-v2",
            "mimo-v2.6-flash-rl-uncensored-exl3-2x": "mimo-v2.6-flash-rl-uncensored-exl3-v1",
            "mimo-v2.6-flash-rl-uncensored-exl3-2x-lmcache": "mimo-v2.6-flash-rl-uncensored-exl3-v1",
            "pd-disagg-mimo-uncensored-exl3-4x": "mimo-v2.6-flash-rl-uncensored-exl3-v1",
            "pd-disagg-mimo-uncensored-exl3-4x-lmcache": "mimo-v2.6-flash-rl-uncensored-exl3-v1",
            "pd-disagg-mimo-exl3-4x": "mimo-v2.6-flash-rl-exl3-v2",
            "pd-disagg-mimo-nvfp4-4x": "mimo-v2.6-flash-nvfp4",
            "mimo-v2.6-flash-rl-nvfp4-2x": "mimo-v2.6-flash-nvfp4",
            "mimo-v2.6-flash-rl-nvfp4-4x": "mimo-v2.6-flash-nvfp4",
            "qwen3-vl-30b-a3b-solo": "qwen3-vl-30b-a3b-instruct-abliterated",
            "4x-spark-cluster/deepseek-v4.1-flash-fulton": "deepseek-v4.1-flash-1017",
            "4x-spark-cluster/minimax-m2.5-fulton": "minimax-m2.5",
            "4x-spark-cluster/qwen3.5-397b-a17B-fp8-fulton": "qwen3.5-397b-a17b-fp8",
        }
        for name in (
            "2x", "2x-store-l2", "2x-store-l2-mtp", "2x-lmcache-l2",
            "4x", "4x-store-l2", "4x-store-l2-mtp",
        ):
            mappings[f"qwen3.8-27b-coder390-{name}"] = "qwen3.8-27b-coder390-ethink-dflash2-nvfp4-w4a16"
        for name, slug in mappings.items():
            with self.subTest(recipe=name):
                recipe = runner.load_recipe(ROOT / f"recipes/{name}.yaml")
                self.assertEqual(recipe["env"]["FES_WEIGHTS_SLUG"], slug)
                self.assertEqual(runner.fes_weights_slug(slug), slug)

    def test_unmapped_4x_recipes_remain_without_guessed_slugs(self):
        # The two upstream 4x recipes serve their HF repo id directly
        # (`vllm serve MiniMaxAI/MiniMax-M2.5`, `vllm serve
        # Qwen/Qwen3.5-397B-A17B-FP8 --load-format instanttensor`) with no
        # mods/fes-weights, so they are hub-cache recipes: the staged contract
        # for those checkpoints belongs to their -fulton twins above, exactly
        # as deepseek-v4.1-flash's does here. Declaring a slug on them would
        # switch the launch to a read-only /opt/llm/models/<slug> mount with
        # HF_HUB_OFFLINE=1 while the command still names the hub id, and the
        # checkpoints are not staged on any rank today. cluster-config still
        # maps both in SPARK_VLLM_DOCKER_SLUGS, so a staged launch remains
        # available per invocation via --mode/--slug; the recipe file simply
        # does not claim a default it cannot satisfy.
        for name in (
            "deepseek-v4.1-flash", "minimax-m2.5", "nemotron-3-ultra-nvfp4",
            "qwen3.5-397b-a17B-fp8", "qwen3.5-397b-int4-autoround",
        ):
            with self.subTest(recipe=name):
                recipe = runner.load_recipe(ROOT / f"recipes/4x-spark-cluster/{name}.yaml")
                self.assertNotIn("FES_WEIGHTS_SLUG", recipe["env"])

    def test_unmapped_fes_recipes_retain_legacy_behavior(self):
        for name in (
            "mimo-v2.6-flash-2x", "mimo-v2.6-flash-4x", "mimo-v2.6-flash-exl3-4x",
            "mimo-v2.6-pro-rl-uncensored-exl3-4x", "3x-spark-cluster/mimo-v2.6-flash-pp3",
            "qwen3.8-flash-next-nvfp4-cluster-fulton",
        ):
            with self.subTest(recipe=name):
                recipe = runner.load_recipe(ROOT / f"recipes/{name}.yaml")
                self.assertIn("mods/fes-weights", recipe["mods"])
                self.assertNotIn("FES_WEIGHTS_SLUG", recipe["env"])

    # The Qwen NVFP4 / QAD arms of the shared recipe<->launcher contract: canonical
    # spec -> the staged-weights fields that recipe must declare. The spec strings
    # are the design doc's corrected placement (.design/spark-vllm-docker/
    # new-qwen-recipes.md:102-112 -- recipes/fulton/<family>/, superseding the plan's
    # recipes/qwen3.8-27b/ and recipes/qwen3.8-flash-next/), and they are the very
    # strings cluster-config keys SPARK_VLLM_DOCKER_SLUGS / _MAX_PARALLEL /
    # _LITELLM_ROUTES by and cluster-control-infra stores in preset.recipe, so this
    # table is the coupling point: a one-sided rename turns a row red.
    #
    # node_count is the launcher's -n arity. Every one of these recipes is
    # cluster_only and declares tensor_parallel=node_count, pipeline_parallel=1, so
    # the launch case below runs in cluster mode and cross-checks that shard count.
    #
    # The parked faithful (worker-cls) QAD pair has NO row here. It is registered in
    # neither the cluster-config maps nor the CCI presets, so a row would be the one
    # place that parked variant still exists in code -- which is how a parked design
    # decision leaks back in.
    QWEN_NVFP4_CONTRACT = {
        "fulton/qwen3.8-27b-nvfp4/qwen3.8-27b-nvfp4-2x": {
            "model": "RedHatAI/Qwen3.8-27B-NVFP4",
            "weights_slug": "qwen3.8-27b-nvfp4",
            "draft_slug": None,
            "draft_hub_model": None,
            "node_count": 2,
        },
        "fulton/qwen3.8-27b-nvfp4/qwen3.8-27b-nvfp4-4x": {
            "model": "RedHatAI/Qwen3.8-27B-NVFP4",
            "weights_slug": "qwen3.8-27b-nvfp4",
            "draft_slug": None,
            "draft_hub_model": None,
            "node_count": 4,
        },
        "fulton/qwen3.8-27b-nvfp4/qwen3.8-27b-nvfp4-2x-draft4b": {
            "model": "RedHatAI/Qwen3.8-27B-NVFP4",
            "weights_slug": "qwen3.8-27b-nvfp4",
            "draft_slug": "qwen3.5-4b-qwen3.8-27b-draft",
            "draft_hub_model": "nvidia/Qwen3.5-4B-Qwen3.8-27B-Draft",
            "node_count": 2,
        },
        "fulton/qwen3.8-27b-nvfp4/qwen3.8-27b-nvfp4-4x-draft4b": {
            "model": "RedHatAI/Qwen3.8-27B-NVFP4",
            "weights_slug": "qwen3.8-27b-nvfp4",
            "draft_slug": "qwen3.5-4b-qwen3.8-27b-draft",
            "draft_hub_model": "nvidia/Qwen3.5-4B-Qwen3.8-27B-Draft",
            "node_count": 4,
        },
        "fulton/qwen3.8-27b-nvfp4/qwen3.8-27b-nvfp4-2x-draft08b": {
            "model": "RedHatAI/Qwen3.8-27B-NVFP4",
            "weights_slug": "qwen3.8-27b-nvfp4",
            "draft_slug": "qwen3.5-0.8b-qwen3.8-27b-draft",
            "draft_hub_model": "nvidia/Qwen3.5-0.8B-Qwen3.8-27B-Draft",
            "node_count": 2,
        },
        "fulton/qwen3.8-27b-nvfp4/qwen3.8-27b-nvfp4-4x-draft08b": {
            "model": "RedHatAI/Qwen3.8-27B-NVFP4",
            "weights_slug": "qwen3.8-27b-nvfp4",
            "draft_slug": "qwen3.5-0.8b-qwen3.8-27b-draft",
            "draft_hub_model": "nvidia/Qwen3.5-0.8B-Qwen3.8-27B-Draft",
            "node_count": 4,
        },
        "fulton/qwen3.8-flash-next-nvfp4/qwen3.8-flash-next-nvfp4-qad-nvidia-2x": {
            "model": "nvidia/Qwen3.8-Flash-Next-NVFP4-QAD",
            "weights_slug": "qwen3.8-flash-next-nvfp4-qad-nvidia",
            "draft_slug": None,
            "draft_hub_model": None,
            "node_count": 2,
        },
        "fulton/qwen3.8-flash-next-nvfp4/qwen3.8-flash-next-nvfp4-qad-nvidia-4x": {
            "model": "nvidia/Qwen3.8-Flash-Next-NVFP4-QAD",
            "weights_slug": "qwen3.8-flash-next-nvfp4-qad-nvidia",
            "draft_slug": None,
            "draft_hub_model": None,
            "node_count": 4,
        },
    }

    # Specs whose recipe file is legitimately absent. Every contract spec has now
    # landed, so this MUST stay empty: declared_recipe() fails outright on a
    # missing file that is not listed here, and
    # test_qwen_nvfp4_unlanded_allowlist_tracks_the_recipe_files fails if a file
    # exists while its spec is still listed. Keeping it empty is enforced by the
    # suite rather than remembered by whoever lands a recipe.
    NOT_YET_LANDED = frozenset()

    def recipe_path(self, spec):
        """The repo path a contract spec's recipe file must occupy."""
        return ROOT / "recipes" / f"{spec}.yaml"

    def declared_recipe(self, spec):
        """The raw (un-defaulted) recipe document a contract row binds to.

        Nothing here may build a recipe from the row it is about to assert against:
        an assertion against a document derived from the row is true by construction
        and keeps passing whether the real file is absent, mis-keyed or renamed. A
        missing file is therefore a hard failure unless the spec is listed in
        NOT_YET_LANDED, where it is reported as a skip instead.
        """
        path = self.recipe_path(spec)
        if not path.is_file():
            if spec in self.NOT_YET_LANDED:
                self.skipTest(f"recipes/{spec}.yaml has not landed yet")
            self.fail(f"recipes/{spec}.yaml is missing, but {spec} is in "
                      "QWEN_NVFP4_CONTRACT")
        with open(path) as handle:
            return runner.yaml.safe_load(handle)

    def test_qwen_nvfp4_unlanded_allowlist_tracks_the_recipe_files(self):
        self.assertLessEqual(self.NOT_YET_LANDED, set(self.QWEN_NVFP4_CONTRACT),
                             "NOT_YET_LANDED names a spec the contract does not")
        for spec in sorted(self.QWEN_NVFP4_CONTRACT):
            with self.subTest(recipe=spec):
                if self.recipe_path(spec).is_file():
                    self.assertNotIn(
                        spec, self.NOT_YET_LANDED,
                        f"recipes/{spec}.yaml now exists; drop it from "
                        "NOT_YET_LANDED so the contract binds to the file")

    def test_qwen_nvfp4_specs_declare_their_drafter_contract(self):
        for spec, row in self.QWEN_NVFP4_CONTRACT.items():
            with self.subTest(recipe=spec):
                recipe = self.declared_recipe(spec)
                weights_slug = (recipe.get("env") or {}).get("FES_WEIGHTS_SLUG")
                self.assertEqual(weights_slug, row["weights_slug"])
                self.assertEqual(recipe.get("model"), row["model"])
                self.assertEqual(
                    runner.fes_weights_slug(row["weights_slug"]), row["weights_slug"])
                # Absence is asserted on the raw document: load_recipe() would have
                # filled the key with its default.
                if row["draft_slug"] is None:
                    self.assertNotIn("draft_slug", recipe)
                    self.assertNotIn("draft_hub_model", recipe)
                    self.assertIn(recipe.get("draft_dir"), (None, "/drafter"))
                    continue
                self.assertEqual(recipe.get("draft_slug"), row["draft_slug"])
                self.assertEqual(recipe.get("draft_hub_model"), row["draft_hub_model"])
                self.assertIn(recipe.get("draft_dir"), (None, "/drafter"))
                self.assertEqual(
                    runner.fes_weights_slug(row["draft_slug"]), row["draft_slug"])

    def test_qwen_nvfp4_launches_mount_only_the_declared_drafter(self):
        for spec, row in self.QWEN_NVFP4_CONTRACT.items():
            with self.subTest(recipe=spec):
                declaration = self.declared_recipe(spec)
                # The row's arity has to be the file's own shard count: -n is how the
                # launcher learns the node count, and every one of these recipes is
                # cluster_only, so --solo would be rejected outright. Topology lives
                # under the recipe's `defaults:` block, which run-recipe.py merges
                # into the command placeholders; there is no top-level
                # `tensor_parallel_size` and no `node_count` key in this schema, so
                # reading those here would compare two identical defaults and pass
                # whatever the shipped file says.
                defaults = declaration.get("defaults") or {}
                self.assertEqual(
                    defaults.get("tensor_parallel", 1)
                    * defaults.get("pipeline_parallel", 1),
                    row["node_count"], f"recipes/{spec}.yaml shard count")
                with tempfile.TemporaryDirectory() as temporary:
                    weights_root = Path(temporary) / "models"
                    for slug in (row["weights_slug"], row["draft_slug"]):
                        if slug is None:
                            continue
                        staged = weights_root / slug
                        staged.mkdir(parents=True)
                        (staged / "config.json").write_text("{}")
                    nodes = ",".join(f"192.0.2.{index}"
                                     for index in range(1, row["node_count"] + 1))
                    # The shipped file is the launch input, so a renamed or dropped
                    # draft_slug in the YAML reaches the resolver here.
                    argv = ["run-recipe.py", str(self.recipe_path(spec)),
                            "--config", "/dev/null", "-n", nodes,
                            "--weights-root", str(weights_root)]
                    launched = {}
                    output = io.StringIO()
                    with patch.object(sys, "argv", argv), \
                            contextlib.redirect_stdout(output), \
                            patch.object(runner, "ENV_FILE", None), \
                            patch.object(runner, "check_model_exists", return_value=True), \
                            patch.object(runner, "check_image_exists", return_value=True), \
                            patch.object(runner, "download_model", return_value=True), \
                            patch.object(runner.subprocess, "run", side_effect=lambda command: (
                                launched.update(args=command)
                                or subprocess.CompletedProcess(command, 0))):
                        self.assertEqual(runner.main(), 0, output.getvalue())
                    args = launched["args"]
                    env = dict(item.split("=", 1)
                               for item in EnvironmentTests.env_args(args))
                    expected = []
                    if row["draft_slug"] is None:
                        self.assertEqual(
                            [key for key in env if key.startswith("FES_DRAFT_")], [])
                    else:
                        draft_dir = declaration.get("draft_dir") or "/drafter"
                        expected = [f"{weights_root / row['draft_slug']}:{draft_dir}:ro"]
                        self.assertEqual(env["FES_DRAFT_SLUG"], row["draft_slug"])
                        self.assertEqual(env["FES_DRAFT_ENABLED"], "1")
                        self.assertEqual(env["FES_DRAFT_DIR"], draft_dir)
                        self.assertEqual(env["FES_DRAFT_HUB_MODEL"],
                                         row["draft_hub_model"])
                    # Exactly one drafter mount, plus the target's own, and no other
                    # staged directory leaks in. Mounts the launcher adds for
                    # unrelated host paths are outside this assertion's scope.
                    staged = [volume for volume in StagedWeightsTests.args_for(args, "-v")
                              if volume.startswith(f"{weights_root}/")]
                    self.assertEqual(sorted(staged), sorted(expected + [
                        f"{weights_root / row['weights_slug']}:/model:ro",
                    ]))


class Coder390ActivationTests(unittest.TestCase):
    def test_activation_is_recipe_owned_and_explicit_env_still_wins(self):
        cases = [
            ("qwen3.8-27b-coder390-2x-store-l2", 2, "MOONCAKE_L2_ENABLED", "mods/mooncake-store-l2"),
            ("qwen3.8-27b-coder390-4x-store-l2", 4, "MOONCAKE_L2_ENABLED", "mods/mooncake-store-l2"),
            ("qwen3.8-27b-coder390-2x-store-l2-mtp", 2, "MOONCAKE_L2_ENABLED", "mods/mooncake-store-l2"),
            ("qwen3.8-27b-coder390-4x-store-l2-mtp", 4, "MOONCAKE_L2_ENABLED", "mods/mooncake-store-l2"),
            ("qwen3.8-27b-coder390-2x-lmcache-l2", 2, "LMCACHE_MP_ENABLED", "mods/lmcache-mp-engine"),
        ]
        for name, node_count, key, mod in cases:
            recipe_path = ROOT / "recipes" / f"{name}.yaml"
            recipe = runner.load_recipe(recipe_path)
            self.assertEqual(recipe["env"][key], "1", name)
            self.assertIn(mod, recipe["mods"], name)
            nodes = ",".join(f"192.0.2.{i}" for i in range(1, node_count + 1))
            for explicit in (False, True):
                with self.subTest(recipe=name, explicit=explicit):
                    flags = ["--default-env", f"{key}=0"]
                    if explicit:
                        flags += ["-e", f"{key}=0"]
                    argv = ["run-recipe.py", str(recipe_path), "--config", "/dev/null",
                            "-n", nodes, *flags]
                    output = io.StringIO()
                    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), \
                            patch.object(runner, "ENV_FILE", None), \
                            patch.object(runner, "check_model_exists", return_value=True), \
                            patch.object(runner, "check_image_exists", return_value=True), \
                            patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as launch:
                        self.assertEqual(runner.main(), 0, output.getvalue())
                    actual = EnvironmentTests.env_args(launch.call_args.args[0])
                    self.assertIn(f"{key}={0 if explicit else 1}", actual)
                    with patch.object(sys, "argv", [*argv, "--dry-run"]), \
                            contextlib.redirect_stdout(io.StringIO()) as dry:
                        self.assertEqual(runner.main(), 0, dry.getvalue())
                    launch_line = next(line for line in dry.getvalue().splitlines()
                                       if line.startswith("   ./launch-cluster.sh"))
                    self.assertEqual(EnvironmentTests.env_args(shlex.split(launch_line.rstrip(" \\"))), actual)

    def test_plain_recipes_do_not_declare_relocation_guard(self):
        for nodes in (2, 4):
            recipe = runner.load_recipe(ROOT / "recipes" / f"qwen3.8-27b-coder390-{nodes}x.yaml")
            self.assertNotIn("MAMBA_SPEC_RELOCATE_GUARD", recipe.get("env") or {})
            self.assertNotIn("mods/mamba-spec-relocate-guard", recipe.get("mods") or [])


if __name__ == "__main__":
    unittest.main()
