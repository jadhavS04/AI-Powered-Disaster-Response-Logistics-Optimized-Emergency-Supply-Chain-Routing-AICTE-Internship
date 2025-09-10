import streamlit as st
import pandas as pd
import joblib
import networkx as nx
from shapely.geometry import Point, LineString
import folium
from streamlit_folium import folium_static
import shapely.wkt
import geopandas as gpd
import heapq
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import KDTree
import numpy as np
import warnings
import json

st.set_page_config(page_title="AI-Powered Disaster Logistics", layout="wide")
st.title("🚑 AI-Powered Disaster Response Logistics")
st.write("Find an optimized route between a selected vehicle and a selected shelter.")

# --- Load Data and Graph ---
@st.cache_resource
def load_data_and_graph():
    try:
        # Load CSVs
        shelters = pd.read_csv("mumbai_shelters_large.csv")
        vehicles = pd.read_csv("mumbai_vehicles_large.csv")
        hazard_zones = pd.read_csv("mumbai_hazard_zones_large.csv")
        roads_nodes = pd.read_csv("mumbai_roads_nodes.csv")
        roads_edges = pd.read_csv("mumbai_roads_edges.csv")

        # Create MultiDiGraph
        G = nx.MultiDiGraph()

        # Add node geometries
        roads_nodes = roads_nodes.reset_index(drop=False).rename(columns={'index':'orig_index'})
        roads_nodes['geometry_point'] = roads_nodes.apply(lambda r: Point(r['x'], r['y']), axis=1)

        # KDTree for nearest node
        node_id_list = roads_nodes.index.to_list()
        node_coords = np.vstack(roads_nodes[["x", "y"]].values)
        kd_tree = KDTree(node_coords)

        def nearest_node_from_coords(coords):
            dist, idx = kd_tree.query([coords[0], coords[1]])
            return node_id_list[int(idx)]

        # Add nodes
        for nid, row in roads_nodes.iterrows():
            G.add_node(nid, x=float(row["x"]), y=float(row["y"]), geometry=row["geometry_point"])

        # Add edges
        for i, row in roads_edges.iterrows():
            try:
                geom_wkt = row.get("geometry", None)
                if not geom_wkt:
                    continue
                edge_geom = shapely.wkt.loads(geom_wkt)
                start_xy = edge_geom.coords[0]
                end_xy = edge_geom.coords[-1]
                u = nearest_node_from_coords(start_xy)
                v = nearest_node_from_coords(end_xy)
                attrs = row.drop(labels=["geometry"], errors="ignore").to_dict()
                attrs["geometry"] = edge_geom
                attrs["length"] = float(attrs.get("length", LineString(edge_geom).length))
                osmid = attrs.get("osmid", None)
                attrs["osmid"] = osmid
                reversed_flag = attrs.get("reversed", False)
                if isinstance(reversed_flag, str):
                    reversed_flag = reversed_flag.lower() in ("true","1","t","yes")
                if reversed_flag:
                    G.add_edge(v,u,**attrs)
                else:
                    G.add_edge(u,v,**attrs)
            except Exception as e:
                warnings.warn(f"Skipping edge at row {i} due to: {e}")
                continue

        # Hazard buffers
        roads_edges_gdf = gpd.GeoDataFrame(roads_edges.copy(), geometry=roads_edges["geometry"].apply(shapely.wkt.loads), crs="EPSG:4326")
        if {"latitude","longitude","risk_level"}.issubset(hazard_zones.columns):
            hazard_gdf = gpd.GeoDataFrame(hazard_zones.copy(), geometry=gpd.points_from_xy(hazard_zones.longitude, hazard_zones.latitude), crs="EPSG:4326")
            try:
                hazard_gdf_m = hazard_gdf.to_crs(epsg=32643)
                high_risk = hazard_gdf_m[hazard_gdf_m["risk_level"].astype(str).str.lower() == "high"]
                high_risk_bufs = high_risk.geometry.buffer(500)
                high_risk_bufs_gdf = gpd.GeoDataFrame(geometry=high_risk_bufs, crs=hazard_gdf_m.crs).to_crs(epsg=4326)
            except Exception:
                high_risk = hazard_gdf[hazard_gdf["risk_level"].astype(str).str.lower() == "high"]
                approx_buffer_deg = 0.005
                high_risk_bufs_gdf = gpd.GeoDataFrame(geometry=high_risk.geometry.buffer(approx_buffer_deg), crs="EPSG:4326")

            try:
                hazardous_roads = gpd.sjoin(roads_edges_gdf, high_risk_bufs_gdf, how="inner", predicate="intersects")
            except TypeError:
                hazardous_roads = gpd.sjoin(roads_edges_gdf, high_risk_bufs_gdf, how="inner", op="intersects")
            hazardous_osmids = set(hazardous_roads["osmid"].dropna().tolist())
        else:
            hazardous_osmids = set()

        # Mark hazards in graph
        for u,v,key,data in G.edges(keys=True,data=True):
            edge_osmid = data.get("osmid", None)
            is_hazard = False
            if edge_osmid is None:
                is_hazard = False
            elif isinstance(edge_osmid,(list,tuple,set,np.ndarray)):
                try:
                    is_hazard = any((str(x) in hazardous_osmids) or (x in hazardous_osmids) for x in edge_osmid)
                except Exception:
                    is_hazard = any(x in hazardous_osmids for x in edge_osmid)
            else:
                is_hazard = (edge_osmid in hazardous_osmids) or (str(edge_osmid) in hazardous_osmids)
            G[u][v][key]["hazard"] = is_hazard

        # Load model
        try:
            best_rf_model = joblib.load("best_random_forest_model.joblib")
            scaler = joblib.load("scaler.joblib")
            le = LabelEncoder()
            le.fit(["Low","Medium","High"])
        except Exception:
            best_rf_model = scaler = le = None

        return {
            "shelters": shelters,
            "vehicles": vehicles,
            "hazard_zones": hazard_zones,
            "G": G,
            "best_rf_model": best_rf_model,
            "scaler": scaler,
            "le": le,
            "kd_tree": kd_tree,
            "node_id_list": node_id_list
        }

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Load data
data = load_data_and_graph()
shelters = data["shelters"]
vehicles = data["vehicles"]
hazard_zones = data["hazard_zones"]
G = data["G"]
best_rf_model = data["best_rf_model"]
scaler = data["scaler"]
le = data["le"]
kd_tree = data["kd_tree"]
node_id_list = data["node_id_list"]

# --- Sidebar UI ---
st.sidebar.header("⚙️ Controls")

# Vehicle selection
available_vehicles = vehicles[vehicles["status"]=="Available"]["vehicle_id"].tolist() if not vehicles.empty else []
selected_vehicle_id = st.sidebar.selectbox("Select Vehicle", available_vehicles)

# Shelter selection
shelter_ids = shelters["shelter_id"].tolist() if not shelters.empty else []
selected_shelter_id = st.sidebar.selectbox("Select Shelter", shelter_ids)

# Hazard penalty slider
hazard_penalty = st.sidebar.slider("Hazard Penalty (cost units)", 100, 5000, 1000, step=100)

# Predict Shelter Risk
if best_rf_model is not None and scaler is not None and le is not None and selected_shelter_id is not None:
    try:
        sel_shelter = shelters[shelters["shelter_id"]==selected_shelter_id].iloc[0]
        shelter_features = pd.DataFrame([[sel_shelter["latitude"], sel_shelter["longitude"], sel_shelter["capacity"]]],
                                        columns=["latitude","longitude","capacity"])
        scaled = scaler.transform(shelter_features)
        pred_enc = best_rf_model.predict(scaled)
        pred_label = le.inverse_transform(pred_enc)[0]
        st.sidebar.markdown(f"### 🛡️ Predicted Risk Level: **{pred_label}**")
    except Exception:
        st.sidebar.write("Prediction not available.")
else:
    st.sidebar.write("Prediction model not loaded.")

# --- Helper Functions ---
def get_nearest_node_from_point(point: Point):
    dist, idx = kd_tree.query([point.x, point.y])
    return node_id_list[int(idx)]

def dijkstra_with_hazards(graph: nx.MultiDiGraph, start_node, end_node, weight='length', hazard_penalty=1000):
    distances = {n: float('inf') for n in graph.nodes()}
    distances[start_node] = 0
    predecessors = {n: None for n in graph.nodes()}
    pq = [(0, start_node)]
    visited = set()
    while pq:
        dist_u, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end_node:
            path=[]
            cur=end_node
            while cur is not None:
                path.append(cur)
                cur = predecessors[cur]
            return path[::-1], distances[end_node]
        for _,v,key,data in graph.edges(u,keys=True,data=True):
            base_w = data.get(weight,1.0)
            if data.get("hazard",False):
                base_w += hazard_penalty
            new_dist = dist_u + base_w
            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = u
                heapq.heappush(pq,(new_dist,v))
    return None,float('inf')

# --- Route Calculation ---
if st.sidebar.button("🚀 Find Optimized Route"):
    if selected_vehicle_id is None or selected_shelter_id is None:
        st.error("Select both vehicle and shelter.")
    else:
        try:
            vehicle_row = vehicles[vehicles["vehicle_id"]==selected_vehicle_id].iloc[0]
            shelter_row = shelters[shelters["shelter_id"]==selected_shelter_id].iloc[0]

            start_point = Point(vehicle_row["longitude"], vehicle_row["latitude"])
            end_point = Point(shelter_row["longitude"], shelter_row["latitude"])
            start_node = get_nearest_node_from_point(start_point)
            end_node = get_nearest_node_from_point(end_point)

            path, cost = dijkstra_with_hazards(G, start_node, end_node, hazard_penalty=hazard_penalty)

            if path:
                st.success(f"✅ Route found! Total cost: **{cost:.2f}**")
                route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]

                # Map
                map_center = [(vehicle_row["latitude"]+shelter_row["latitude"])/2,
                              (vehicle_row["longitude"]+shelter_row["longitude"])/2]
                m = folium.Map(location=map_center, zoom_start=12)
                folium.Marker([vehicle_row["latitude"], vehicle_row["longitude"]], popup=f"🚗 Vehicle {selected_vehicle_id}", icon=folium.Icon(color="green",icon="car")).add_to(m)
                folium.Marker([shelter_row["latitude"], shelter_row["longitude"]], popup=f"🏠 Shelter {selected_shelter_id}", icon=folium.Icon(color="blue",icon="home")).add_to(m)

                # Hazard markers
                color_map = {"Low":"green","Medium":"orange","High":"red"}
                for _, rz in hazard_zones.iterrows():
                    try:
                        folium.CircleMarker([float(rz["latitude"]), float(rz["longitude"])], radius=6,
                                            color=color_map.get(str(rz.get("risk_level")),"gray"),
                                            fill=True, fill_opacity=0.6,
                                            popup=f"Hazard Zone {rz.get('zone_id','')} ({rz.get('risk_level','')})").add_to(m)
                    except:
                        continue

                # Draw route
                folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.9).add_to(m)

                folium_static(m, width=1000, height=600)

                # Download route
                route_df = pd.DataFrame(route_coords, columns=["lat","lon"])
                st.download_button("⬇️ Download Route CSV", route_df.to_csv(index=False), "route.csv", "text/csv")
                st.download_button("⬇️ Download Route GeoJSON", json.dumps({
                    "type":"LineString",
                    "coordinates":[(lon,lat) for lat,lon in route_coords]
                }), "route.geojson", "application/geo+json")

            else:
                st.error("❌ No path found.")
        except Exception as e:
            st.error(f"Error during routing: {e}")
