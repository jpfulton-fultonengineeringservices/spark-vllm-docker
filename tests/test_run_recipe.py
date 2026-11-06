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
            "4x-spark-cluster/minimax-m2.5": "minimax-m2.5",
            "4x-spark-cluster/qwen3.5-397b-a17B-fp8-fulton": "qwen3.5-397b-a17b-fp8",
            "4x-spark-cluster/qwen3.5-397b-a17B-fp8": "qwen3.5-397b-a17b-fp8",
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
        for name in (
            "deepseek-v4.1-flash", "nemotron-3-ultra-nvfp4",
            "qwen3.5-397b-int4-autoround",
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
