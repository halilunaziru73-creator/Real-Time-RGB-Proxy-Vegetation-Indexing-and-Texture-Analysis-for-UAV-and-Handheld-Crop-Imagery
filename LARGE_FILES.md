# Large Files Not Included in This Push

Three items from your original uploads could **not** be pushed here because this
environment's network cannot reach GitHub's LFS storage backend
(`github-cloud.s3.amazonaws.com`), and one of them also exceeds GitHub's flat
100MB-per-file limit:

| File | Size | Why it's excluded |
|---|---|---|
| `build/main/main.pkg` | 288MB | Exceeds GitHub's 100MB limit; requires Git LFS |
| `build/main/PYZ-00.pyz` | 60MB | Requires Git LFS (bundled with the above) |
| `dist/main.exe` | 289MB | Exceeds GitHub's 100MB limit; requires Git LFS |
| `GACL_Data.xlsx` | 13.6MB | Under 100MB, but was bundled with LFS tracking for consistency |

## How to add them yourself

From a machine with normal internet access (not this sandboxed one):

```bash
git clone https://github.com/halilunaziru73-creator/Real-Time-RGB-Proxy-Vegetation-Indexing-and-Texture-Analysis-for-UAV-and-Handheld-Crop-Imagery.git
cd Real-Time-RGB-Proxy-Vegetation-Indexing-and-Texture-Analysis-for-UAV-and-Handheld-Crop-Imagery

# Install Git LFS if you don't have it: https://git-lfs.github.com/
git lfs install

# Track the large file types
git lfs track "*.pkg" "*.pyz" "*.exe" "*.xlsx"
git add .gitattributes

# Copy in the files you want (build/, dist/, GACL_Data.xlsx) then:
git add build dist GACL_Data.xlsx
git commit -m "Add build artifacts and reference dataset via Git LFS"
git push
```

**Note on GitHub's free LFS quota:** the free tier includes 1GB of storage and
1GB of bandwidth per month. `main.pkg` + `PYZ-00.pyz` + `main.exe` + `GACL_Data.xlsx`
together total ~650MB, which would consume most of that in one push. Consider
whether the compiled build artifacts genuinely need to live in version control,
or whether a GitHub Release attachment (no LFS needed, no quota impact) would
suit better — see below.

## Alternative: GitHub Releases (recommended for the compiled binaries)

Compiled/packaged outputs like `main.exe` and the `build/` folder are usually a
better fit for a **GitHub Release** than for LFS-tracked repo files, since Release
assets don't count against LFS quota:

```bash
gh release create v1.0.0 dist/main.exe build/main/main.pkg build/main/PYZ-00.pyz \
  --title "N_GACL v1.0.0" --notes "Packaged Windows build"
```
