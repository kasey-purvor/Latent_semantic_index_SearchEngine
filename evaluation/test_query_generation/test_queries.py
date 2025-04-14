"""
Test queries for the latent semantic index search engine.
This module contains a collection of test queries organized by topic representation.
"""

test_queries = {
    "queries": [
        # Higher-Representation Topics
        {
            "id": 1,
            "query": "Social theory impact on political discourse",
            "topic_id": 14,
            "topic_name": "Social & Political Theory",
            "topic_representation": 12.2,
            "topic_keywords": ["article", "social", "paper", "new", "theory", "way", "political", "work", "issue", "one"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        {
            "id": 2,
            "query": "Contemporary societal issues in academic literature",
            "topic_id": 14,
            "topic_name": "Social & Political Theory",
            "topic_representation": 12.2,
            "topic_keywords": ["article", "social", "paper", "new", "theory", "way", "political", "work", "issue", "one"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        {
            "id": 3,
            "query": "Theoretical frameworks for analyzing social phenomena",
            "topic_id": 14,
            "topic_name": "Social & Political Theory",
            "topic_representation": 12.2,
            "topic_keywords": ["article", "social", "paper", "new", "theory", "way", "political", "work", "issue", "one"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        {
            "id": 4,
            "query": "Computational modeling approaches for complex systems",
            "topic_id": 2,
            "topic_name": "Methods & Models",
            "topic_representation": 12.0,
            "topic_keywords": ["model", "method", "using", "system", "result", "data", "based", "time", "used", "approach"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        {
            "id": 5,
            "query": "Data-driven methodologies in scientific research",
            "topic_id": 2,
            "topic_name": "Methods & Models",
            "topic_representation": 12.0,
            "topic_keywords": ["model", "method", "using", "system", "result", "data", "based", "time", "used", "approach"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        {
            "id": 6,
            "query": "Time-based analysis techniques for large datasets",
            "topic_id": 2,
            "topic_name": "Methods & Models",
            "topic_representation": 12.0,
            "topic_keywords": ["model", "method", "using", "system", "result", "data", "based", "time", "used", "approach"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        {
            "id": 7,
            "query": "Technology integration in educational systems",
            "topic_id": 11,
            "topic_name": "Learning & Development",
            "topic_representation": 9.6,
            "topic_keywords": ["paper", "learning", "system", "research", "development", "information", "design", "use", "management", "technology"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        {
            "id": 8,
            "query": "Knowledge management framework development",
            "topic_id": 11,
            "topic_name": "Learning & Development",
            "topic_representation": 9.6,
            "topic_keywords": ["paper", "learning", "system", "research", "development", "information", "design", "use", "management", "technology"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        {
            "id": 9,
            "query": "Machine learning applications in information systems",
            "topic_id": 11,
            "topic_name": "Learning & Development",
            "topic_representation": 9.6,
            "topic_keywords": ["paper", "learning", "system", "research", "development", "information", "design", "use", "management", "technology"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        {
            "id": 10,
            "query": "Evidence-based policy making in European governance",
            "topic_id": 4,
            "topic_name": "Policy & Economics",
            "topic_representation": 9.5,
            "topic_keywords": ["policy", "economic", "european", "state", "country", "evidence", "government", "public", "market", "paper"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        {
            "id": 11,
            "query": "Economic implications of state market regulations",
            "topic_id": 4,
            "topic_name": "Policy & Economics",
            "topic_representation": 9.5,
            "topic_keywords": ["policy", "economic", "european", "state", "country", "evidence", "government", "public", "market", "paper"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        {
            "id": 12,
            "query": "Public sector reform strategies in national economies",
            "topic_id": 4,
            "topic_name": "Policy & Economics",
            "topic_representation": 9.5,
            "topic_keywords": ["policy", "economic", "european", "state", "country", "evidence", "government", "public", "market", "paper"],
            "representation_category": "high",
            "is_interdisciplinary": False
        },
        
        # Moderate-Representation Topics
        {
            "id": 13,
            "query": "Community-based interventions for child health services",
            "topic_id": 5,
            "topic_name": "Health & Social Care",
            "topic_representation": 6.7,
            "topic_keywords": ["health", "social", "care", "study", "child", "service", "people", "community", "school", "research"],
            "representation_category": "moderate",
            "is_interdisciplinary": False
        },
        {
            "id": 14,
            "query": "Social determinants of healthcare outcomes",
            "topic_id": 5,
            "topic_name": "Health & Social Care",
            "topic_representation": 6.7,
            "topic_keywords": ["health", "social", "care", "study", "child", "service", "people", "community", "school", "research"],
            "representation_category": "moderate",
            "is_interdisciplinary": False
        },
        {
            "id": 15,
            "query": "Laser threshold characteristics in semiconductor devices",
            "topic_id": 12,
            "topic_name": "Optics & Devices",
            "topic_representation": 5.7,
            "topic_keywords": ["none", "laser", "book", "dot", "device", "recombination", "threshold", "carrier", "state", "nucleus"],
            "representation_category": "moderate",
            "is_interdisciplinary": False
        },
        {
            "id": 16,
            "query": "Carrier recombination mechanisms in quantum dot structures",
            "topic_id": 12,
            "topic_name": "Optics & Devices",
            "topic_representation": 5.7,
            "topic_keywords": ["none", "laser", "book", "dot", "device", "recombination", "threshold", "carrier", "state", "nucleus"],
            "representation_category": "moderate",
            "is_interdisciplinary": False
        },
        {
            "id": 17,
            "query": "Concentration effects on reaction rates",
            "topic_id": 16,
            "topic_name": "Effects & Studies",
            "topic_representation": 5.3,
            "topic_keywords": ["effect", "study", "using", "rate", "sample", "result", "concentration", "different", "found", "level"],
            "representation_category": "moderate",
            "is_interdisciplinary": False
        },
        {
            "id": 18,
            "query": "Sample size influence on experimental results",
            "topic_id": 16,
            "topic_name": "Effects & Studies",
            "topic_representation": 5.3,
            "topic_keywords": ["effect", "study", "using", "rate", "sample", "result", "concentration", "different", "found", "level"],
            "representation_category": "moderate",
            "is_interdisciplinary": False
        },
        {
            "id": 19,
            "query": "Cost-effectiveness of clinical treatment options",
            "topic_id": 15,
            "topic_name": "Medical Research",
            "topic_representation": 4.7,
            "topic_keywords": ["patient", "study", "health", "cost", "risk", "treatment", "group", "trial", "data", "clinical"],
            "representation_category": "moderate",
            "is_interdisciplinary": False
        },
        {
            "id": 20,
            "query": "Quantum properties of optical materials at varying temperatures",
            "topic_id": 18,
            "topic_name": "Physics & Properties",
            "topic_representation": 4.0,
            "topic_keywords": ["energy", "temperature", "surface", "using", "structure", "property", "quantum", "optical", "measurement", "state"],
            "representation_category": "moderate",
            "is_interdisciplinary": False
        },
        
        # Lower-Representation Topics
        {
            "id": 21,
            "query": "Magnetic field interactions with electron clusters",
            "topic_id": 1,
            "topic_name": "Fields & Observations",
            "topic_representation": 3.8,
            "topic_keywords": ["wave", "field", "magnetic", "observation", "observed", "cluster", "electron", "frequency", "wind", "region"],
            "representation_category": "low",
            "is_interdisciplinary": False
        },
        {
            "id": 22,
            "query": "Global warming impacts on polar ice formation",
            "topic_id": 8,
            "topic_name": "Climate & Environment",
            "topic_representation": 3.7,
            "topic_keywords": ["change", "ice", "climate", "data", "sediment", "sea", "southern", "antarctic", "model", "within"],
            "representation_category": "low",
            "is_interdisciplinary": False
        },
        {
            "id": 23,
            "query": "Visual object recognition in neural networks",
            "topic_id": 10,
            "topic_name": "Visual Perception",
            "topic_representation": 3.2,
            "topic_keywords": ["task", "object", "visual", "network", "group", "information", "result", "representation", "two", "experiment"],
            "representation_category": "low",
            "is_interdisciplinary": False
        },
        {
            "id": 24,
            "query": "Boundary layer effects on fluid motion and vortex formation",
            "topic_id": 9,
            "topic_name": "Fluid Dynamics",
            "topic_representation": 2.5,
            "topic_keywords": ["flow", "boundary", "game", "velocity", "pressure", "motion", "auroral", "vortex", "video", "fluid"],
            "representation_category": "low",
            "is_interdisciplinary": False
        },
        {
            "id": 25,
            "query": "Population growth patterns in early South African history",
            "topic_id": 7,
            "topic_name": "History & Demographics",
            "topic_representation": 2.4,
            "topic_keywords": ["growth", "south", "early", "life", "history", "war", "african", "africa", "population", "year"],
            "representation_category": "low",
            "is_interdisciplinary": False
        },
        {
            "id": 26,
            "query": "Wage competition effects on urban housing markets",
            "topic_id": 3,
            "topic_name": "Markets & Economics",
            "topic_representation": 2.3,
            "topic_keywords": ["market", "price", "wage", "employee", "competition", "exchange", "contract", "equilibrium", "housing", "urban"],
            "representation_category": "low",
            "is_interdisciplinary": False
        },
        {
            "id": 27,
            "query": "Genetic expression mechanisms in human cancer tissue",
            "topic_id": 6,
            "topic_name": "Genetics & Biology",
            "topic_representation": 2.2,
            "topic_keywords": ["cell", "human", "gene", "dna", "cancer", "genetic", "expression", "tissue", "available", "population"],
            "representation_category": "low",
            "is_interdisciplinary": False
        },
        {
            "id": 28,
            "query": "Protein receptor binding and activation pathways",
            "topic_id": 0,
            "topic_name": "Molecular Biology",
            "topic_representation": 2.0,
            "topic_keywords": ["response", "protein", "receptor", "effect", "role", "binding", "mechanism", "activation", "rat", "membrane"],
            "representation_category": "low",
            "is_interdisciplinary": False
        },
        {
            "id": 29,
            "query": "Groundwater contamination sources and environmental impact",
            "topic_id": 13,
            "topic_name": "Environmental Science",
            "topic_representation": 1.9,
            "topic_keywords": ["water", "soil", "environmental", "land", "risk", "concentration", "source", "groundwater", "carbon", "impact"],
            "representation_category": "low",
            "is_interdisciplinary": False
        },
        {
            "id": 30,
            "query": "Material stress patterns in thin film applications",
            "topic_id": [17, 19],
            "topic_name": "Interdisciplinary (Materials & Patterns)",
            "topic_representation": [1.7, 1.5],
            "topic_keywords": ["stress", "film", "published", "available", "food", "material", "paper", "thin", "journal", "contact", 
                             "pattern", "state", "movement", "reaction", "trajectory", "behavior", "forecast", "image", "event", "chaotic"],
            "representation_category": "low",
            "is_interdisciplinary": True
        }
    ],
    "metadata": {
        "total_queries": 30,
        "representation_distribution": {
            "high": 12,
            "moderate": 8,
            "low": 10
        },
        "topic_distribution": {
            "14": {"name": "Social & Political Theory", "representation": 12.2, "query_count": 3},
            "2": {"name": "Methods & Models", "representation": 12.0, "query_count": 3},
            "11": {"name": "Learning & Development", "representation": 9.6, "query_count": 3},
            "4": {"name": "Policy & Economics", "representation": 9.5, "query_count": 3},
            "5": {"name": "Health & Social Care", "representation": 6.7, "query_count": 2},
            "12": {"name": "Optics & Devices", "representation": 5.7, "query_count": 2},
            "16": {"name": "Effects & Studies", "representation": 5.3, "query_count": 2},
            "15": {"name": "Medical Research", "representation": 4.7, "query_count": 1},
            "18": {"name": "Physics & Properties", "representation": 4.0, "query_count": 1},
            "1": {"name": "Fields & Observations", "representation": 3.8, "query_count": 1},
            "8": {"name": "Climate & Environment", "representation": 3.7, "query_count": 1},
            "10": {"name": "Visual Perception", "representation": 3.2, "query_count": 1},
            "9": {"name": "Fluid Dynamics", "representation": 2.5, "query_count": 1},
            "7": {"name": "History & Demographics", "representation": 2.4, "query_count": 1},
            "3": {"name": "Markets & Economics", "representation": 2.3, "query_count": 1},
            "6": {"name": "Genetics & Biology", "representation": 2.2, "query_count": 1},
            "0": {"name": "Molecular Biology", "representation": 2.0, "query_count": 1},
            "13": {"name": "Environmental Science", "representation": 1.9, "query_count": 1},
            "17": {"name": "Materials Science", "representation": 1.7, "query_count": 0.5},
            "19": {"name": "Patterns & Behavior", "representation": 1.5, "query_count": 0.5}
        },
        "interdisciplinary_queries": 1,
        "stratification_method": "Proportional to topic representation with at least one query per topic"
    }
}

# Convenience functions to access the data
def get_all_queries():
    """Return all test queries."""
    return test_queries["queries"]

def get_queries_by_category(category):
    """Return queries filtered by representation category (high, moderate, low)."""
    return [q for q in test_queries["queries"] if q["representation_category"] == category]

def get_queries_by_topic_id(topic_id):
    """Return queries for a specific topic ID."""
    return [q for q in test_queries["queries"] if q["topic_id"] == topic_id or 
            (isinstance(q["topic_id"], list) and topic_id in q["topic_id"])]

def get_metadata():
    """Return metadata about the query collection."""
    return test_queries["metadata"]

# If the file is run directly
if __name__ == "__main__":
    # Print some basic information about the test queries
    print(f"Total queries: {len(test_queries['queries'])}")
    print(f"Categories: {test_queries['metadata']['representation_distribution']}")
    print(f"Topics: {len(test_queries['metadata']['topic_distribution'])}")
