# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
GIS Data Search Example — VecForge

Demonstrates building a geospatial data discovery system with:
- Indexing GIS dataset descriptions and metadata
- Filtering by data format, resolution, and region
- Semantic search for spatial dataset discovery
- Namespace isolation per data provider

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

Usage:
    python examples/gis_data_search.py
"""

from __future__ import annotations

from vecforge import VecForge


def main() -> None:
    """Run the GIS data search demo."""

    db = VecForge(":memory:")

    print("🌍 VecForge GIS Data Search Demo")
    print("=" * 50)

    # ─── Set up provider namespaces ───
    db.create_namespace("usgs")
    db.create_namespace("copernicus")
    db.create_namespace("openstreetmap")

    # ─── Add GIS datasets ───
    datasets = [
        # USGS datasets
        ("SRTM Digital Elevation Model — 30m resolution global elevation data "
         "derived from NASA Shuttle Radar Topography Mission. Coverage: 60°N to 56°S. "
         "Format: GeoTIFF. Useful for terrain analysis, slope calculation, and "
         "hydrological modelling. CRS: EPSG:4326 (WGS84).",
         {"format": "GeoTIFF", "resolution": "30m", "region": "global",
          "type": "DEM", "year": 2020, "crs": "EPSG:4326"},
         "usgs"),

        ("NLCD Land Cover Dataset — 30m resolution land cover classification "
         "for the contiguous United States. 16 land cover classes including "
         "forest, agriculture, urban, water, and wetlands. Updated annually. "
         "Format: GeoTIFF. CRS: Albers Equal Area (EPSG:5070).",
         {"format": "GeoTIFF", "resolution": "30m", "region": "usa",
          "type": "land_cover", "year": 2024, "crs": "EPSG:5070"},
         "usgs"),

        ("USGS Earthquake Hazard Map — Seismic hazard assessment for the US. "
         "Peak ground acceleration (PGA) values at 2% probability of exceedance "
         "in 50 years. Vector polygons with hazard classifications. "
         "Format: Shapefile/GeoJSON. CRS: EPSG:4326.",
         {"format": "GeoJSON", "resolution": "variable", "region": "usa",
          "type": "hazard_map", "year": 2025, "crs": "EPSG:4326"},
         "usgs"),

        # Copernicus / ESA datasets
        ("Sentinel-2 Multispectral Imagery — 10m resolution optical satellite "
         "imagery with 13 spectral bands. Global coverage with 5-day revisit. "
         "Applications: vegetation monitoring, NDVI, land use change detection, "
         "crop health. Format: JPEG2000/Cloud Optimized GeoTIFF.",
         {"format": "COG", "resolution": "10m", "region": "global",
          "type": "satellite_imagery", "year": 2026, "crs": "EPSG:32632"},
         "copernicus"),

        ("Copernicus DEM GLO-30 — Global 30m Digital Elevation Model derived "
         "from TanDEM-X SAR interferometry. Higher accuracy than SRTM in "
         "mountainous regions. Format: Cloud Optimized GeoTIFF. "
         "CRS: EPSG:4326. Vertical accuracy: <4m (LE90).",
         {"format": "COG", "resolution": "30m", "region": "global",
          "type": "DEM", "year": 2024, "crs": "EPSG:4326"},
         "copernicus"),

        ("CORINE Land Cover — 100m resolution land cover/land use map for "
         "Europe with 44 thematic classes organised in 3 hierarchical levels. "
         "Minimum mapping unit: 25 hectares. Format: GeoPackage/Shapefile. "
         "CRS: ETRS89-LAEA (EPSG:3035). Updated every 6 years.",
         {"format": "GeoPackage", "resolution": "100m", "region": "europe",
          "type": "land_cover", "year": 2024, "crs": "EPSG:3035"},
         "copernicus"),

        ("Sentinel-1 SAR Backscatter data — C-band Synthetic Aperture Radar "
         "imagery for all-weather, day-and-night Earth observation. 5x20m "
         "resolution. Applications: flood mapping, soil moisture, deforestation, "
         "ship detection. Format: GeoTIFF. Dual polarisation: VV+VH.",
         {"format": "GeoTIFF", "resolution": "10m", "region": "global",
          "type": "SAR", "year": 2026, "crs": "EPSG:4326"},
         "copernicus"),

        # OpenStreetMap derived datasets
        ("OpenStreetMap Road Network — Global road network vector data "
         "extracted from OSM. Includes highway classification, road names, "
         "speed limits, surface type, and lane counts. Format: GeoJSON/Shapefile. "
         "CRS: EPSG:4326. Updated daily.",
         {"format": "GeoJSON", "resolution": "vector", "region": "global",
          "type": "road_network", "year": 2026, "crs": "EPSG:4326"},
         "openstreetmap"),

        ("OSM Building Footprints — Global building polygon dataset with height "
         "information where available. Includes building type, levels, roof shape. "
         "Applications: urban planning, population estimation, shadow analysis. "
         "Format: GeoJSON. CRS: EPSG:4326.",
         {"format": "GeoJSON", "resolution": "vector", "region": "global",
          "type": "building_footprints", "year": 2026, "crs": "EPSG:4326"},
         "openstreetmap"),

        ("OSM Water Bodies — Rivers, lakes, reservoirs, and coastlines from "
         "OpenStreetMap. Includes waterway classification, name, width, and "
         "seasonal attributes. Used for hydrological analysis and flood risk. "
         "Format: GeoPackage. CRS: EPSG:4326.",
         {"format": "GeoPackage", "resolution": "vector", "region": "global",
          "type": "hydrology", "year": 2026, "crs": "EPSG:4326"},
         "openstreetmap"),
    ]

    print("\n🗺️  Adding GIS datasets...")
    for text, metadata, namespace in datasets:
        doc_id = db.add(text, metadata=metadata, namespace=namespace)
        print(f"  ✅ [{namespace}] {metadata['type']}: {doc_id[:8]}...")

    # ─── Search Examples ───

    # 1. Find elevation / terrain data
    print("\n\n🔍 Search: 'elevation terrain DEM' (all providers)")
    print("-" * 50)
    results = db.search("elevation terrain digital elevation model", top_k=3)
    for r in results:
        print(f"  [{r.score:.4f}] [{r.namespace}] {r.metadata.get('type')} — "
              f"{r.metadata.get('resolution')} — {r.text[:60]}...")

    # 2. Search for vegetation monitoring data
    print("\n\n🔍 Search: 'vegetation NDVI crop monitoring satellite'")
    print("-" * 50)
    results = db.search("vegetation NDVI crop monitoring satellite imagery", top_k=3)
    for r in results:
        print(f"  [{r.score:.4f}] [{r.namespace}] {r.metadata.get('type')} — "
              f"{r.text[:70]}...")

    # 3. Filter by format and resolution
    print("\n\n🔍 Search: High resolution datasets in GeoTIFF format")
    print("-" * 50)
    results = db.search(
        "high resolution spatial data",
        filters={"format": {"in": ["GeoTIFF", "COG"]}},
        top_k=5,
    )
    for r in results:
        print(f"  [{r.score:.4f}] {r.metadata.get('format')} — "
              f"{r.metadata.get('resolution')} — {r.text[:50]}...")

    # 4. Copernicus-only search
    print("\n\n🔍 Search: 'flood mapping radar' (Copernicus only)")
    print("-" * 50)
    results = db.search(
        "flood mapping radar all weather imaging",
        namespace="copernicus",
        top_k=3,
    )
    for r in results:
        print(f"  [{r.score:.4f}] [{r.namespace}] {r.metadata.get('type')} — "
              f"{r.text[:70]}...")

    # 5. Land use / land cover search
    print("\n\n🔍 Search: 'land use land cover classification'")
    print("-" * 50)
    results = db.search(
        "land use land cover classification map",
        filters={"type": "land_cover"},
        top_k=3,
    )
    for r in results:
        print(f"  [{r.score:.4f}] [{r.namespace}] Region: {r.metadata.get('region')} — "
              f"CRS: {r.metadata.get('crs')} — {r.text[:50]}...")

    # 6. Infrastructure data from OSM
    print("\n\n🔍 Search: 'road network routing navigation' (OSM only)")
    print("-" * 50)
    results = db.search(
        "road network infrastructure routing navigation",
        namespace="openstreetmap",
        top_k=3,
    )
    for r in results:
        print(f"  [{r.score:.4f}] {r.metadata.get('type')} — {r.text[:70]}...")

    # ─── Stats ───
    print("\n\n📊 GIS Catalog Statistics")
    print("-" * 50)
    stats = db.stats()
    print(f"  Total datasets: {stats['documents']}")
    for ns, count in stats["namespace_counts"].items():
        if ns != "default":
            print(f"  {ns}: {count} datasets")

    db.close()
    print("\n✅ GIS demo complete! VecForge by Suneel Bose K · ArcGX TechLabs")


if __name__ == "__main__":
    main()
