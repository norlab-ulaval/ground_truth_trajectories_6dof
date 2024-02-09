import cartopy.crs as ccrs
from cartopy.io import img_tiles
from pyproj import Proj
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#map_generation(east_0, north_0, 20, 50, 10, 10, 50)
def world_map(east, north, tiler_size, width_left, width_right, height_down, height_up, data, rovers,point_scatter) :
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection= ccrs.epsg(2949))
    tiler = img_tiles.GoogleTiles(desired_tile_form='RGB', style = 'satellite')
    ax.add_image(tiler, tiler_size, interpolation='spline36')
    ax.set_xlabel("East")
    ax.set_ylabel("North")  
    extent = [east - width_left, east + width_right, north - height_down, north + height_up]  
    ax.set_extent(extent, crs = ccrs.epsg(2949))
    ax.plot(east, north, 'ro', transform = ccrs.epsg(2949))
    for rover, data in rovers.items():
    # ax.scatter(data['East'], data['North'], label=rover, alpha=0.8, s=3, transform=ccrs.epsg(2949))
        ax.plot(data['East'], data['North'], label=rover, alpha=0.8, transform=ccrs.epsg(2949))
        ax.scatter(data['East'][point_scatter],data['North'][point_scatter])
    ax.set_title("GNSS Rovers Coordinates with Google Tiles")
    ax.legend()
    plt.show()