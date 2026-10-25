"""CLI for the host-staged Mooncake store connector installer.

Subcommands mirror the house mod CLI shape (detect|install|verify|selftest):

    install   install the vendored store package over the image's copy
    verify    confirm the installed state is the vendored one
    selftest  offline integrity checks (no vLLM required)
    manifest  print the gate constants (for run.sh / CI parity)
"""

from __future__ import annotations

import argparse
import json
import sys

from . import envs_shim, gates, installer


def _cmd_install(args: argparse.Namespace) -> int:
    try:
        status = installer.install(args.site, apply_envs_shim=not args.no_envs_shim)
    except (installer.InstallError, gates.GateError, envs_shim.EnvShimError) as exc:
        print(f"[mooncake-worker] FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"[mooncake-worker] OK: {status}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        status = installer.verify(args.site)
    except (installer.InstallError, gates.GateError) as exc:
        print(f"[mooncake-worker] FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"[mooncake-worker] OK: {status}")
    return 0


def _cmd_install_p2p(args: argparse.Namespace) -> int:
    try:
        status = installer.install_p2p(args.site)
        envs_py = installer.SitePaths.resolve(args.site).envs_py
        if envs_py.is_file():
            added = envs_shim.inject_p2p_envs(envs_py)
            if added:
                status += f"; injected {added} P2P envs"
    except (installer.InstallError, gates.GateError, envs_shim.EnvShimError) as exc:
        print(f"[mooncake-worker] FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"[mooncake-worker] OK: {status}")
    return 0


def _cmd_verify_p2p(args: argparse.Namespace) -> int:
    try:
        status = installer.verify_p2p(args.site)
        envs_py = installer.SitePaths.resolve(args.site).envs_py
        if envs_py.is_file():
            envs_shim.verify_p2p_envs(envs_py)
    except (installer.InstallError, gates.GateError, envs_shim.EnvShimError) as exc:
        print(f"[mooncake-worker] FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"[mooncake-worker] OK: {status}")
    return 0


def _cmd_selftest(_args: argparse.Namespace) -> int:
    rc = 0
    rc |= gates._selftest()
    rc |= envs_shim._selftest()
    try:
        gates.verify_p2p_vendor()
    except gates.GateError as exc:
        print(f"[mooncake-worker] SELFTEST FAIL (p2p vendor): {exc}", file=sys.stderr)
        rc |= 1
    return 1 if rc else 0


def _cmd_manifest(_args: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "md5_stock_workers": list(gates.MD5_STOCK_WORKERS),
                "md5_mod_worker": gates.MD5_MOD_WORKER,
                "package_files": list(gates.PACKAGE_FILES),
                "p2p_package_files": list(gates.P2P_PACKAGE_FILES),
                "md5_mod_p2p_connector": gates.MD5_MOD_P2P_CONNECTOR,
                "md5_stock_p2p_connectors": list(gates.MD5_STOCK_P2P_CONNECTORS),
                "envs_target": envs_shim.TARGET_ENV,
                "vendor_dir": str(gates.vendor_dir()),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mooncake-worker", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_install = sub.add_parser("install", help="install the vendored store package")
    p_install.add_argument("--site", default=installer.VLLM_SITE_DEFAULT)
    p_install.add_argument(
        "--no-envs-shim",
        action="store_true",
        help="skip the vllm/envs.py VLLM_PREFIX_CACHE_RETENTION_INTERVAL inject",
    )
    p_install.set_defaults(func=_cmd_install)

    p_verify = sub.add_parser("verify", help="verify the installed state")
    p_verify.add_argument("--site", default=installer.VLLM_SITE_DEFAULT)
    p_verify.set_defaults(func=_cmd_verify)

    p_install_p2p = sub.add_parser(
        "install-p2p", help="install the vendored P2P MooncakeConnector package"
    )
    p_install_p2p.add_argument("--site", default=installer.VLLM_SITE_DEFAULT)
    p_install_p2p.set_defaults(func=_cmd_install_p2p)

    p_verify_p2p = sub.add_parser("verify-p2p", help="verify the installed P2P package + envs")
    p_verify_p2p.add_argument("--site", default=installer.VLLM_SITE_DEFAULT)
    p_verify_p2p.set_defaults(func=_cmd_verify_p2p)

    sub.add_parser("selftest", help="offline integrity checks").set_defaults(func=_cmd_selftest)
    sub.add_parser("manifest", help="print gate constants").set_defaults(func=_cmd_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
