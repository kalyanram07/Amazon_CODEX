"""
Creates the official submission ZIP archive conforming to Amazon ML Challenge specifications.
"""
import os
import zipfile
import sys


def build_submission_zip(
    zip_name: str = "CODEX_submission.zip",
    base_dir: str = ".",
):
    print(f"Building submission zip: {zip_name}...")

    required_files = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("Documentation_template.md", "Documentation_template.md"),
    ]

    for src_rel, _ in required_files:
        full_src = os.path.join(base_dir, src_rel)
        if not os.path.exists(full_src):
            print(f"Error: Required file missing: {full_src}")
            return False

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        for src_rel, arc_name in required_files:
            full_src = os.path.join(base_dir, src_rel)
            print(f"  Adding {src_rel} -> {arc_name}")
            zf.write(full_src, arc_name)

        # 2. Code artefacts
        code_base = os.path.join(base_dir, "code", "business_entity_resolution")
        for root, dirs, files in os.walk(code_base):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                print(f"  Adding {rel_path}")
                zf.write(full_path, rel_path)

    zip_size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"\n[SUCCESS] Successfully created {zip_name} ({zip_size_mb:.2f} MB)")
    return True


if __name__ == "__main__":
    success = build_submission_zip()
    sys.exit(0 if success else 1)
