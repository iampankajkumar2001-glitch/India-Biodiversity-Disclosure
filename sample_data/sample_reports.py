"""
sample_data/sample_reports.py
------------------------------
Three short, entirely FICTIONAL company-report excerpts used for the
in-app "Try sample data" demo mode, so you can see the pipeline run
end-to-end before uploading real filings. None of these refer to real
companies — they are illustrative stand-ins written to exercise the
positive / negative / regulation sentence types the way the paper's
Appendix Table A.4 examples do for US 10-Ks.

Keyed as "Company Name|Year" -> report text.
"""

SAMPLE_REPORTS: dict[str, str] = {
    "Nilgiri Estates Beverages Ltd|2023": """
        Nilgiri Estates Beverages Ltd sources tea and coffee from estates
        bordering the Western Ghats, a recognised biodiversity hotspot.
        We have committed to a deforestation-free supply chain by 2026 and
        work with estate partners to protect shade-grown forest cover that
        supports local wildlife corridors. Our biodiversity stewardship
        program has restored over 40 hectares of degraded catchment area
        adjacent to our estates, improving habitat connectivity for
        endemic bird species. We believe these ecosystem services,
        including watershed protection and pollination, are directly
        linked to the long-term yield and quality of our tea gardens.
        The company continues to comply with the Wildlife Protection Act
        and state forest regulations governing land bordering reserved
        forest areas. In FY23, no instances of non-compliance with
        biodiversity-related regulation were reported. We are proud that
        our afforestation and agroforestry initiatives have been
        recognised by the state biodiversity board as a model for
        sustainable estate management.
    """,
    "Deccan Ferro Mining Corporation|2022": """
        Our iron ore mining operations in central India are located near
        forest areas that provide habitat for several threatened species,
        including elephants that use a historic elephant corridor
        bordering our lease area. Expansion of the mine has required
        clearance under the Forest Conservation Act and an environmental
        impact assessment addressing potential habitat fragmentation.
        Overburden disposal and tailings management pose physical risks
        to downstream wetland ecosystems and freshwater availability for
        surrounding communities. During the year, the National Green
        Tribunal directed the company to submit an updated biodiversity
        management plan after concerns were raised about encroachment
        into a wildlife sanctuary buffer zone. Dust and discharge from
        processing units have been linked by local groups to declining
        fish populations in the adjoining river system. Litigation
        relating to forest land diversion remains pending before the
        state High Court. We recognise that continued biodiversity loss
        in the region could result in further regulatory restrictions on
        our mining licence and increased compliance costs going forward.
    """,
    "GreenVolt Renewable Power Ltd|2023": """
        GreenVolt operates wind and solar assets across coastal and
        semi-arid regions of India. Site selection for new wind farms
        requires careful assessment of migratory bird flight paths, as
        turbine blades can pose a collision risk to raptors and other
        avian species protected under wildlife regulations. Several
        proposed project sites near the coast fall within the Coastal
        Regulation Zone, requiring additional clearances before
        construction can begin. We conduct pre-construction biodiversity
        surveys and post-construction monitoring of bird and bat
        mortality at all wind sites, and have re-routed access roads at
        two locations to avoid disturbance to a nearby wetland habitat.
        Community consultations flagged concerns about the loss of
        grazing land and disruption to a sacred grove near one proposed
        site, and the project layout was subsequently revised. GreenVolt
        also discloses ongoing engagement with the state forest
        department regarding land use approvals for transmission
        corridors that pass through reserved forest.
    """,
}
