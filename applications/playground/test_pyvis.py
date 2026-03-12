from pyvis.network import Network

net = Network(height="1080px", width="100%", bgcolor="#222222", font_color="white")

# Make the node ACTUALLY be the image
net.add_node(
    "hello",
    label="Research Note",
    shape="image",
    image="/home/chuang/hcg/projects/control_suite/temp/lab_downstairs_test_2/rgb/rgb_1770050781381919648.png",
)  # Use your local path here too

net.add_node(2, label="Connecting Idea")
net.add_edge("hello", 2)

net.show("image_nodes.html", notebook=False)
