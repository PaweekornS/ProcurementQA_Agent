#!/usr/bin/env python3
"""
scripts/export_to_neo4j.py

Migrates the cached Thai Government Procurement Knowledge Graph from
InMemoryGraphDB (outputs/openrouter_graph_db.pkl) into a Neo4j database.

Features:
- Batched UNWIND ingestion for high throughput.
- Strips heavy embedding vectors to optimize Neo4j memory and visual rendering.
- Robust label mapping (:Cases, :Laws, :Crimes, :Cluster).
- Works with or without APOC plugin.
- Includes a --dry-run mode for testing data sanitization without an active Neo4j instance.
"""

import os
import sys
import pickle
import argparse
from typing import Dict, Any, List
from collections import Counter
from dotenv import load_dotenv

# Try importing neo4j driver
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


def sanitize_value(val: Any) -> Any:
    """Sanitize properties to valid Neo4j primitive types."""
    if val is None:
        return ""
    if isinstance(val, (int, float, bool, str)):
        return val
    if isinstance(val, (list, tuple, set)):
        clean_list = []
        for x in val:
            if isinstance(x, (int, float, bool, str)):
                clean_list.append(x)
            else:
                clean_list.append(str(x))
        return clean_list
    if isinstance(val, dict):
        return str(val)
    return str(val)


def sanitize_node_properties(node_id: str, raw_data: Dict[str, Any], node_type: str) -> Dict[str, Any]:
    """Sanitize node attributes and drop raw embedding vectors."""
    clean = {
        "id": str(node_id),
        "node_type": node_type
    }
    for k, v in raw_data.items():
        if k == "embedding":
            continue  # Exclude dense vectors to keep UI light & fast
        clean[k] = sanitize_value(v)
    return clean


def sanitize_edge_properties(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize relationship attributes."""
    clean = {}
    for k, v in raw_data.items():
        if k == "relation_type":
            continue
        clean[k] = sanitize_value(v)
    return clean


def parse_args():
    parser = argparse.ArgumentParser(description="Export Knowledge Graph to Neo4j")
    parser.add_argument(
        "--input", "-i",
        default=os.getenv("graph_db_path", "outputs/openrouter_graph_db.pkl"),
        help="Path to openrouter_graph_db.pkl (default: outputs/openrouter_graph_db.pkl)"
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j Bolt URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD", "procurement123"),
        help="Neo4j password (default: procurement123)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for Cypher UNWIND transactions (default: 500)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing database contents before ingestion"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and sanitize graph without connecting to Neo4j"
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[-] Error: Graph pickle not found at: {input_path}")
        sys.exit(1)

    print(f"[*] Loading knowledge graph from: {input_path}")
    with open(input_path, "rb") as f:
        kg_data = pickle.load(f)

    nx_graph = kg_data["graph"]
    nodes_data = kg_data.get("nodes_data", {})
    embeddings = kg_data.get("embeddings", {})

    print(f"[+] Loaded NetworkX Graph:")
    print(f"    - Nodes: {nx_graph.number_of_nodes():,}")
    print(f"    - Edges: {nx_graph.number_of_edges():,}")
    print(f"    - Embedding Keys: {list(embeddings.keys())}")

    # Inspect node types
    type_counts = Counter()
    categorized_nodes = {
        "Cases": [],
        "Laws": [],
        "Crimes": [],
        "Cluster": [],
        "Entity": []
    }

    for node_id, attrs in nx_graph.nodes(data=True):
        node_info = nodes_data.get(node_id, {})
        ntype = attrs.get("node_type") or node_info.get("type", "Entity")
        type_counts[ntype] += 1

        raw_props = node_info.get("data", {}).copy()
        clean_props = sanitize_node_properties(node_id, raw_props, ntype)

        target_bucket = ntype if ntype in categorized_nodes else "Entity"
        categorized_nodes[target_bucket].append(clean_props)

    print("\n[*] Node Distribution:")
    for ntype, count in type_counts.most_common():
        print(f"    - {ntype}: {count:,}")

    # Inspect edge types
    rel_counts = Counter()
    categorized_edges = {}

    for u, v, edge_data in nx_graph.edges(data=True):
        rel_type = edge_data.get("relation_type", "RELATED_TO")
        clean_rel = rel_type.strip().upper().replace(" ", "_").replace("-", "_")
        rel_counts[clean_rel] += 1

        clean_props = sanitize_edge_properties(edge_data)
        if clean_rel not in categorized_edges:
            categorized_edges[clean_rel] = []

        categorized_edges[clean_rel].append({
            "source": str(u),
            "target": str(v),
            "props": clean_props
        })

    print("\n[*] Relationship Distribution:")
    for rel_type, count in rel_counts.most_common():
        print(f"    - [:{rel_type}]: {count:,}")

    if args.dry_run:
        print("\n[OK] Dry-run completed successfully! Graph data is well-formed.")
        return

    # Check neo4j package
    if not NEO4J_AVAILABLE:
        print("\n[-] Error: The 'neo4j' package is not installed.")
        print("    Please run: pip install neo4j>=5.15.0")
        sys.exit(1)

    print(f"\n[*] Connecting to Neo4j at {args.uri}...")
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    try:
        with driver.session() as session:
            # Check APOC availability
            has_apoc = False
            try:
                apoc_check = session.run("RETURN apoc.version() AS v").single()
                if apoc_check:
                    has_apoc = True
                    print(f"[+] APOC plugin detected (version: {apoc_check['v']})")
            except Exception:
                print("[-] APOC plugin not available; using native Cypher UNWIND.")

            if args.clear:
                print("[!] Clearing existing graph database...")
                session.run("MATCH (n) DETACH DELETE n")
                print("[+] Cleared database.")

            # Create Constraints / Indexes
            print("[*] Creating uniqueness constraints on :Node(id) and specific labels...")
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
            except Exception as e:
                print(f"    - Note for Node constraint: {e}")

            labels_to_constrain = ["Cases", "Laws", "Crimes", "Cluster", "Entity"]
            for label in labels_to_constrain:
                try:
                    session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
                except Exception as e:
                    print(f"    - Note for {label} constraint: {e}")

            # 1. Ingest Nodes by Label
            print("\n[*] Ingesting Nodes into Neo4j...")
            for label, nodes_list in categorized_nodes.items():
                if not nodes_list:
                    continue
                print(f"    - Ingesting {len(nodes_list):,} {label} nodes...")
                for i in range(0, len(nodes_list), args.batch_size):
                    batch = nodes_list[i : i + args.batch_size]
                    query = f"""
                    UNWIND $batch AS item
                    MERGE (n:Node {{id: item.id}})
                    SET n:{label}, n += item
                    """
                    session.run(query, batch=batch)

            # 2. Ingest Relationships
            print("\n[*] Ingesting Relationships into Neo4j...")
            for rel_type, edges_list in categorized_edges.items():
                if not edges_list:
                    continue
                print(f"    - Ingesting {len(edges_list):,} [:{rel_type}] edges...")
                for i in range(0, len(edges_list), args.batch_size):
                    batch = edges_list[i : i + args.batch_size]
                    query = f"""
                    UNWIND $batch AS edge
                    MATCH (s:Node {{id: edge.source}})
                    MATCH (t:Node {{id: edge.target}})
                    MERGE (s)-[r:{rel_type}]->(t)
                    SET r += edge.props
                    """
                    session.run(query, batch=batch)

        print("\n=========================================================")
        print("[SUCCESS] Graph export to Neo4j completed!")
        print("=========================================================")
        print(f"• Neo4j Browser UI:  http://localhost:7474")
        print(f"• Connection Bolt:   {args.uri}")
        print("\nQuick Exploration Cypher Queries to try in the Neo4j Browser:")
        print("---------------------------------------------------------")
        print("1. Overview of Node & Edge types:")
        print("   CALL db.schema.visualization()")
        print("")
        print("2. Explore Thai Procurement Laws and their connected Cases/Crimes:")
        print("   MATCH (l:Laws)-[r]-(target)")
        print("   RETURN l, r, target LIMIT 100")
        print("")
        print("3. Explore Intra-document Statutory Citation Structure:")
        print("   MATCH (l1:Laws)-[r:ADJACENT_SECTION|CITES_CLAUSE]->(l2:Laws)")
        print("   RETURN l1, r, l2 LIMIT 50")
        print("")
        print("4. Find High-Degree Landmark Laws:")
        print("   MATCH (l:Laws)-[r]-()")
        print("   RETURN l.id, l.entry, count(r) AS degree")
        print("   ORDER BY degree DESC LIMIT 15")
        print("=========================================================")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
