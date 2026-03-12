import dash
from dash import html, dcc, Input, Output, State
from pyvis.network import Network
import flask
import os
import uuid  # Used to force the Iframe to refresh

# 1. SETUP PATHS
IMAGE_DIR = "/home/chuang/hcg/projects/control_suite/temp/lab_downstairs_test_2/rgb/"
# Create assets folder if it doesn't exist
if not os.path.exists("assets"):
    os.makedirs("assets")

app = dash.Dash(__name__, assets_folder="assets")


# 2. IMAGE SERVER (Local Images)
@app.server.route("/my_images/<path:path>")
def serve_image(path):
    return flask.send_from_directory(IMAGE_DIR, path)


# 3. DASH INTERFACE
app.layout = html.Div(
    [
        # 1. Header (Fixed height)
        html.Div(
            [
                html.H3("Obsidian-Pyvis Hybrid", style={"color": "white", "margin": "0"}),
                dcc.Input(id="img-name", type="text", placeholder="rgb_1...png", style={"padding": "10px"}),
                html.Button("Add", id="btn-add", n_clicks=0, style={"margin-left": "10px"}),
            ],
            style={"height": "7vh", "padding": "10px", "backgroundColor": "#1a1a1a"},
        ),
        # 2. Container for the Iframe (Must have a height!)
        html.Iframe(
            id="graph-frame",
            # srcDoc will be injected here by the callback
            style={
                "width": "100%",
                "height": "80vh",  # Fill the parent Div
                "border": "none",
            },
        ),
        # Hidden store to hold the clicked node ID
        dcc.Store(id="clicked-node-store"),
        # Hidden elements to bridge JS to Python
        html.Div(
            [
                dcc.Input(id="hidden-input", value=""),
                html.Button(id="hidden-trigger-btn", n_clicks=0),
            ],
            style={"display": "none"},
        ),  # Hide the whole container
        # SIDEBAR (Where the click result shows)
        html.Div(
            [
                html.H4("Node Info", style={"color": "#7e22ce"}),
                html.Div(id="click-output", children="Click a node to see details", style={"color": "white"}),
            ],
            style={
                "height": "8vh",
                "min-height": "50px",
                "width": "100%",
                "padding": "10px",
                "backgroundColor": "#1a1a1a",
                "borderRight": "1px solid #333",
            },
        ),
    ],
    style={"height": "100vh", "margin": "0", "overflow": "hidden"},
)  # The Root container must be 100vh
# Storage for our nodes
nodes_list = []


# 4. CALLBACK: Generate Pyvis HTML and Update Iframe
@app.callback(
    Output("graph-frame", "srcDoc"),  # Using srcDoc is safer than src for 404s
    Input("btn-add", "n_clicks"),
    State("img-name", "value"),
)
def update_pyvis_graph(n_clicks, img_name):
    # Setup Pyvis
    net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white")

    # Add a root node if it's the first run
    if not nodes_list:
        nodes_list.append({"id": "center", "label": "Vault Root"})

    # Add new node from user input
    if img_name:
        node_id = f"node_{n_clicks}"
        # We use the full URL for the image so the Iframe can find it
        img_url = f"http://127.0.0.1:8050/my_images/{img_name}"
        nodes_list.append({"id": node_id, "label": img_name, "image": img_url})

    # Build the network
    for node in nodes_list:
        if "image" in node:
            net.add_node(node["id"], label=node["label"], shape="circularImage", image=node["image"], size=30)
        else:
            net.add_node(node["id"], label=node["label"], color="#7e22ce")

    # Connect everything to center
    for node in nodes_list:
        if node["id"] != "center":
            net.add_edge(node["id"], "center", color="#555")

    # Obsidian-like Physics
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100)

    # Return the HTML content directly to the Iframe to avoid 404 file issues
    custom_html = net.generate_html()
    # Inside your callback, after generating custom_html:
    # We use a very aggressive injection to ensure it's placed after Pyvis initializes
    click_js = """
    <script type="text/javascript">
        // Wait for the network to be fully defined
        setTimeout(function() {
            if (typeof network !== 'undefined') {
                network.on("click", function (params) {
                    if (params.nodes.length > 0) {
                        var nodeId = params.nodes[0];
                        window.parent.postMessage({type: 'node_click', id: nodeId}, "*");
                    }
                });
            }
        }, 500);
    </script>
    """
    custom_html = custom_html.replace("</body>", click_js + "</body>")
    return custom_html


@app.callback(
    Output("click-output", "children"), Input("hidden-trigger-btn", "n_clicks"), State("hidden-input", "value")
)
def handle_pyvis_click(n_clicks, node_id):
    if n_clicks > 0:
        return f"You clicked on: {node_id}. Python can now process this!"
    return "Click a node in the graph..."


app.clientside_callback(
    """
    function(n_clicks) {
        if (!window.hasNodeListener) {
            window.addEventListener("message", (event) => {
                if (event.data.type === 'node_click') {
                    const input = document.getElementById('hidden-input');
                    const btn = document.getElementById('hidden-trigger-btn');
                    
                    // Force the value into the input so Dash sees it
                    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                    setter.call(input, event.data.id);
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    btn.click();
                    console.log("Caught click in Dash:", event.data.id);
                }
            }, false);
            window.hasNodeListener = true;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("hidden-input", "id"),  # Dummy output to satisfy Dash
    Input("graph-frame", "id"),  # Trigger when the graph loads
)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
