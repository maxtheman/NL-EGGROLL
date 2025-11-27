"""
Best-effort NAX availability probe using Metal device capabilities (no private MLX API).
- Prints device name and supported Apple GPU families.
- Heuristic: NAX likely available on M3/A17+ (Apple GPU family >= 9) and recent OS.
"""

import platform

try:
    import Metal  # type: ignore
except Exception as e:
    print("Metal PyObjC bindings not available:", e)
    raise SystemExit(1)


def main():
    dev = Metal.MTLCreateSystemDefaultDevice()
    if dev is None:
        print("No Metal device found")
        return

    name = dev.name()
    print(f"Metal device: {name}")

    # Gather supported Apple GPU families
    families = []
    # Apple families are typically 1..X; iterate a reasonable range.
    for fam in range(1, 16):
        sel = getattr(Metal, f"MTLGPUFamilyApple{fam}", None)
        if sel is None:
            continue
        if dev.supportsFamily_(sel):
            families.append(fam)
    if families:
        print(f"Supported Apple GPU families: {families}")
        max_fam = max(families)
    else:
        print("No Apple GPU family support reported.")
        max_fam = -1

    mac_ver = platform.mac_ver()[0]
    print(f"macOS version: {mac_ver}")

    # Heuristic: treat family >=9 as NAX-capable (M3/A17+ class).
    nax_likely = max_fam >= 9
    print(f"NAX likely available (heuristic, family>=9): {nax_likely}")


if __name__ == "__main__":
    main()
