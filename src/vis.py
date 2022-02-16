import plotly.graph_objects as go

#visualize pointclouds
#pointcloud: shape(minibatch, 3)
def visualize_pointclouds(pointcloud):
  points = pointcloud.points_packed()
  fig = go.Figure(data=[go.Scatter3d(
      x=points[:,0],
      y=points[:,1],
      z=points[:,2],
  )])
  return fig