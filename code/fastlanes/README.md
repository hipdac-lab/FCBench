# FastLanes Benchmark

**TL;DR**  
This directory benchmarks the **FC Bench** dataset with **FastLanes**.  
Run `make run` to download the data, convert, compress every file, validate round-trips, and obtain per-dataset and overall compression-ratio statistics.

---

## Why FastLanes?

FastLanes uses **ALP: Adaptive Lossless Floating-Point Compression** to deliver state-of-the-art, lossless compression for floating-point data.

| Resource               | Link                                                      |
|------------------------|-----------------------------------------------------------|
| ALP paper (SIGMOD ’24) | [doi:10.1145/3626717](https://doi.org/10.1145/3626717)    |
| ALP code               | [cwida/ALP](https://github.com/cwida/ALP)                 |
| FastLanes File Format  | [cwida/fastlanes](https://github.com/cwida/fastlanes)     |

---

## Quick Start

```bash
# from the repository root
cd code/fastlanes
make run  # installs deps, downloads data, runs full benchmark
````

---

## Workflow

1. **Download**

    * Fetches two Google Drive folders (if not already on disk):

        * `HPC_TS_OBS` (time-series observations)
        * `DB` (database-style test files)
2. **Convert**

    * Finds every memory-mapped binary ending in `_f32` or `_f64`.
    * Exports it as a single-column CSV (no header) plus a matching JSON schema.
3. **Compress & Validate**

    * Uses **FastLanes v0.1.4** to compress each CSV → `.fls` and then decompress back to CSV.
    * Verifies bit-exact equality at float32/float64 precision.
    * Writes a `compression_stats.csv` per dataset with sizes, ratio, and a *validated* flag.
4. **Aggregate**

    * Merges all stats into `combined_compression_stats.csv`.
    * Prints average ratio per dataset and an overall mean.
5. **Polished Logging**

    * Every console line is prefixed with `--`; status lines are green, summaries blue; ratios shown with two decimals.

---

## Results (FastLanes v0.1.4)

| Dataset                              | Avg Ratio |
| ------------------------------------ | --------- |
| DB\_tpcds\_catalog\_f32              | 1.12×     |
| DB\_tpcds\_store\_f32                | 1.07×     |
| DB\_tpcds\_web\_f32                  | 1.12×     |
| DB\_tpch\_lineitem\_f32              | 1.51×     |
| HPC\_TS\_OBS\_acs\_wht\_f32          | 1.33×     |
| HPC\_TS\_OBS\_astro\_mhd\_f64        | 21.45×    |
| HPC\_TS\_OBS\_astro\_pt\_f64         | 1.12×     |
| HPC\_TS\_OBS\_citytemp\_f32          | 3.08×     |
| HPC\_TS\_OBS\_g24\_78\_usb2\_f32     | 1.15×     |
| HPC\_TS\_OBS\_h3d\_temp\_f32         | 1.00×     |
| HPC\_TS\_OBS\_hdr\_night\_f32        | 2.97×     |
| HPC\_TS\_OBS\_hdr\_palermo\_f32      | 3.22×     |
| HPC\_TS\_OBS\_hst\_wfc3\_ir\_f32     | 1.50×     |
| HPC\_TS\_OBS\_hst\_wfc3\_uvis\_f32   | 1.47×     |
| HPC\_TS\_OBS\_jane\_street\_f64      | 1.10×     |
| HPC\_TS\_OBS\_jw\_mirimage\_f32      | 1.25×     |
| HPC\_TS\_OBS\_miranda3d\_f32         | 1.34×     |
| HPC\_TS\_OBS\_msg\_bt\_f64           | 1.07×     |
| HPC\_TS\_OBS\_num\_brain\_f64        | 1.15×     |
| HPC\_TS\_OBS\_num\_control\_f64      | 1.11×     |
| HPC\_TS\_OBS\_nyc\_taxi2015\_f64     | 1.77×     |
| HPC\_TS\_OBS\_phone\_gyro\_f64       | 3.49×     |
| HPC\_TS\_OBS\_rsim\_f32              | 1.42×     |
| HPC\_TS\_OBS\_solar\_wind\_f32       | 1.40×     |
| HPC\_TS\_OBS\_spain\_gas\_price\_f64 | 6.42×     |
| HPC\_TS\_OBS\_spitzer\_irac\_f32     | 1.21×     |
| HPC\_TS\_OBS\_ts\_gas\_f32           | 2.44×     |
| HPC\_TS\_OBS\_turbulence\_f32        | 1.17×     |
| HPC\_TS\_OBS\_wave\_f32              | 1.12×     |
| HPC\_TS\_OBS\_wesad\_chest\_f64      | 2.87×     |

**Overall average compression ratio:** **2.45×**
