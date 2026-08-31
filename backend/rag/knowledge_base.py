"""
backend/rag/knowledge_base.py
==============================
Static satellite domain knowledge used to seed the ChromaDB vector store.

Each document is a self-contained paragraph that captures one conceptual
unit of expert knowledge relevant to Landsat 8/9 thermal analysis,
aleatoric uncertainty interpretation, and urban/vegetation land-use
operational assessment.

These documents are embedded once at agent initialisation and retrieved
at query time to ground Granite's briefing generation in factual context.
"""

from __future__ import annotations

from typing import List

# ---------------------------------------------------------------------------
# Document corpus
# ---------------------------------------------------------------------------
# Each entry is a dict with keys:
#   "id"      — stable document identifier (used as ChromaDB doc id)
#   "text"    — the paragraph to embed
#   "metadata"— filterable key-value tags

KNOWLEDGE_DOCUMENTS: List[dict] = [

    # ── Landsat 8/9 Thermal Band ─────────────────────────────────────────────
    {
        "id": "landsat_st_b10_spec",
        "text": (
            "Landsat 8 and 9 Band 10 (ST_B10) is the primary thermal infrared "
            "surface temperature band with a spatial resolution of 100 metres "
            "resampled to 30 metres in Collection 2 Level-2 products. It measures "
            "top-of-atmosphere brightness temperature in Kelvin using a single "
            "detector spanning 10.6–11.19 µm. Digital Number (DN) values range "
            "from approximately 7,500 to 65,455 for typical Earth surface temperatures. "
            "The radiometric scale factor is 0.00341802 Kelvin/DN with an additive "
            "offset of 149.0 K. Accurate surface temperature retrieval requires "
            "atmospheric correction and emissivity estimation."
        ),
        "metadata": {"domain": "sensor", "band": "ST_B10", "satellite": "Landsat8_9"},
    },
    {
        "id": "landsat_collection2_qa",
        "text": (
            "Landsat Collection 2 Level-2 QA_PIXEL band encodes per-pixel quality "
            "flags as bit fields in a 16-bit integer. Bit 3 (cloud) and bit 4 "
            "(cloud shadow) are the most operationally critical: when set, the "
            "corresponding pixels are unreliable for thermal or reflectance analysis. "
            "Bit 1 (dilated cloud) indicates a 3-pixel buffer around detected clouds. "
            "Scenes with cloud cover exceeding 20% should be treated with caution; "
            "above 50% cloud cover the scene is generally considered unsuitable for "
            "quantitative surface temperature retrieval."
        ),
        "metadata": {"domain": "quality", "band": "QA_PIXEL", "satellite": "Landsat8_9"},
    },
    {
        "id": "landsat_emissivity",
        "text": (
            "Surface emissivity is the ratio of actual thermal radiance to blackbody "
            "radiance at the same temperature. For Landsat ST_B10, emissivity is "
            "derived from the ASTER Global Emissivity Dataset (GED) or estimated from "
            "NDVI-based land cover. Urban impervious surfaces typically have emissivities "
            "of 0.95–0.97. Dense vegetation canopy ranges from 0.97 to 0.99. Water "
            "bodies are close to 0.99. Bare soil and desert range from 0.90 to 0.96. "
            "Emissivity errors of ±0.01 translate to surface temperature errors of "
            "approximately ±0.5 K at 300 K surface temperatures."
        ),
        "metadata": {"domain": "physics", "topic": "emissivity"},
    },

    # ── Thermal Urban Analysis ───────────────────────────────────────────────
    {
        "id": "urban_heat_island",
        "text": (
            "The Urban Heat Island (UHI) effect causes city centres to record surface "
            "temperatures 2–10 K above surrounding rural or vegetated areas in Landsat "
            "thermal imagery. Dense building clusters, road networks with low albedo, "
            "and reduced evapotranspiration from impervious surfaces are primary drivers. "
            "Daytime UHI intensity is strongest in summer and during low-wind, clear-sky "
            "conditions. In Landsat ST_B10 imagery UHI zones appear as bright (high-DN) "
            "anomalies in urban cores. These zones typically exhibit moderate MC-Dropout "
            "uncertainty in RGB reconstruction due to the unusual thermal signature "
            "compared to rural training samples."
        ),
        "metadata": {"domain": "urban", "topic": "UHI"},
    },
    {
        "id": "urban_building_detection",
        "text": (
            "Building detection in reconstructed RGB imagery derived from thermal inputs "
            "exploits the thermal mass contrast between rooftop materials and surroundings. "
            "Flat concrete and metal roofs retain heat at night and appear as warm anomalies. "
            "YOLOv8 can detect building outlines in RGB reconstructions, but confidence "
            "should be interpreted cautiously when aleatoric uncertainty in the bounding-box "
            "region exceeds 0.4, as the RGB texture in high-uncertainty zones is speculative. "
            "Detection classes relevant to urban analysis include: building, car, truck, "
            "bus, and road markings."
        ),
        "metadata": {"domain": "urban", "topic": "building_detection"},
    },
    {
        "id": "road_network_thermal",
        "text": (
            "Road networks exhibit distinct thermal signatures due to asphalt's high heat "
            "capacity and low albedo. In Landsat ST_B10 imagery, major arterials appear as "
            "linear warm features during daytime acquisition. In RGB reconstructions from "
            "thermal inputs, roads manifest as grey linear structures with low spectral "
            "variation. YOLOv8 detects road markings and vehicles on roads. Road detections "
            "with adjusted confidence below 0.3 after uncertainty penalisation indicate "
            "the underlying thermal gradient is ambiguous and the road boundary localisation "
            "should be treated as approximate."
        ),
        "metadata": {"domain": "infrastructure", "topic": "roads"},
    },

    # ── Vegetation & Forest ───────────────────────────────────────────────────
    {
        "id": "vegetation_canopy_thermal",
        "text": (
            "Vegetation canopy, particularly dense broadleaf forest, appears as cool "
            "thermal anomalies in Landsat ST_B10 imagery due to high evapotranspiration "
            "rates and canopy shading. Canopy temperatures are typically 3–8 K below "
            "adjacent urban or bare-soil surfaces in midday acquisitions. In RGB "
            "reconstructions from thermal inputs, canopy areas appear as dark-green "
            "or olive-toned patches. Conifer forests show less diurnal temperature "
            "variation than deciduous canopy. MC-Dropout uncertainty is generally "
            "lower over homogeneous forest canopy than over fragmented urban-rural "
            "boundaries."
        ),
        "metadata": {"domain": "vegetation", "topic": "canopy_thermal"},
    },
    {
        "id": "ndvi_thermal_relationship",
        "text": (
            "Normalized Difference Vegetation Index (NDVI) derived from Landsat OLI "
            "bands 4 (Red) and 5 (NIR) is strongly negatively correlated with surface "
            "temperature in ST_B10. Pixels with NDVI > 0.4 typically correspond to "
            "surface temperatures 4–12 K below bare urban surfaces. NDVI < 0.2 indicates "
            "sparse or no vegetation. This relationship underpins split-window algorithms "
            "for emissivity correction. In thermal-to-RGB reconstruction pipelines, "
            "vegetation confidence is highest when the thermal signal is consistently cool "
            "and spatially homogeneous, correlating with low MC-Dropout variance."
        ),
        "metadata": {"domain": "vegetation", "topic": "NDVI"},
    },

    # ── MC-Dropout Uncertainty Interpretation ────────────────────────────────
    {
        "id": "mc_dropout_uncertainty_interpretation",
        "text": (
            "MC-Dropout aleatoric uncertainty in thermal-to-RGB neural reconstruction "
            "quantifies spatial regions where the model produces inconsistent colour "
            "predictions across stochastic forward passes. Uncertainty values normalised "
            "to [0, 1] can be interpreted as: 0.0–0.2 = high fidelity, RGB reconstruction "
            "is stable and features are reliable; 0.2–0.4 = moderate uncertainty, features "
            "are likely correct but spatial boundaries may be slightly offset; 0.4–0.6 = "
            "elevated uncertainty, colour assignments are speculative, detection bounding "
            "boxes should be treated as approximate; > 0.6 = high uncertainty, this region "
            "has poor thermal-to-visual correspondence and any detected features should be "
            "flagged for human review before operational use."
        ),
        "metadata": {"domain": "model", "topic": "uncertainty_interpretation"},
    },
    {
        "id": "adjusted_confidence_interpretation",
        "text": (
            "Adjusted detection confidence is computed as the product of raw YOLO detection "
            "confidence and the complement of mean aleatoric uncertainty over the bounding-box "
            "region: Adjusted Confidence = Raw Confidence × (1.0 - Mean Uncertainty). This "
            "penalises detections whose spatial footprint overlaps high-uncertainty zones. "
            "Adjusted confidence thresholds for operational use: > 0.6 = act on detection; "
            "0.4–0.6 = flag for secondary verification; 0.2–0.4 = treat as an indicator only; "
            "< 0.2 = insufficient evidence, do not use as sole basis for decisions."
        ),
        "metadata": {"domain": "model", "topic": "adjusted_confidence"},
    },

    # ── Operational Guidance ──────────────────────────────────────────────────
    {
        "id": "operational_high_uncertainty_zones",
        "text": (
            "High-uncertainty spatial zones in thermal imagery analysis arise primarily "
            "from: cloud or cloud-shadow contamination (QA_PIXEL bits 3–4), thermal "
            "boundary regions between land-cover classes (mixed pixels at 30m resolution), "
            "water bodies with specular reflectance anomalies, and fire or industrial heat "
            "sources that push thermal values outside the model's training distribution. "
            "Analysts should always cross-reference high-uncertainty detections with the "
            "raw QA_PIXEL mask and, where available, co-registered optical imagery from "
            "the same overpass. Operational decisions based solely on reconstructed RGB "
            "in high-uncertainty zones carry significant epistemic risk."
        ),
        "metadata": {"domain": "operational", "topic": "high_uncertainty"},
    },
    {
        "id": "tactical_urban_forest_profiles",
        "text": (
            "Tactical land-use profiles for urban and forest environments in satellite "
            "analysis: Urban cores exhibit high surface temperature variability (σ > 2 K), "
            "dense detection returns (buildings, vehicles, roads), and moderate-to-high "
            "MC-Dropout uncertainty at urban-rural transition zones. Forest environments "
            "show low surface temperature variability (σ < 1 K) over dense canopy, sparse "
            "non-vegetation detections, and low MC-Dropout uncertainty over homogeneous "
            "canopy interior. Mixed urban-forest fringe zones are the most uncertainty-prone "
            "and require the most conservative interpretation. Riparian corridors along "
            "water bodies are frequently misclassified due to mixed thermal signatures."
        ),
        "metadata": {"domain": "operational", "topic": "land_use_profiles"},
    },
]
