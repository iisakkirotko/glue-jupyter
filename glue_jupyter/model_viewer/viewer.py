import os
from pathlib import Path

import ipyreact
import ipywidgets as widgets
import ipyvuetify as v
import pyvista as pv
import traitlets
from glue_jupyter.ipyvolume.scatter.layer_artist import Scatter3DLayerState

from glue_jupyter.registries import viewer_registry

from numpy import array, clip, isnan, ones, sqrt
from glue_jupyter.state_traitlets_helpers import GlueState

from glue.config import colormaps
from glue_jupyter.view import IPyWidgetView
from glue.core.data import Subset
from glue.viewers.common.layer_artist import LayerArtist
from glue.viewers.scatter.state import ScatterLayerState

from glue_jupyter.widgets import Color, Size
from glue_jupyter.common.state3d import ViewerState3D
from ..link import link, on_change
from ..vuetify_helpers import link_glue_choices


# def export_meshes(meshes, output_path):
#     plotter = pv.Plotter()
#     for info in meshes.values():
#         plotter.add_mesh(info["mesh"], color=info["color"], name=info["name"], opacity=info["opacity"])

#     # TODO: What's the correct way to deal with this?
#     if output_path.endswith(".obj"):
#         plotter.export_obj(output_path)
#     elif output_path.endswith(".gltf"):
#         plotter.export_gltf(output_path)
#     else:
#         raise ValueError("Unsupported extension!")


# TODO: Worry about efficiency later
# and just generally make this better
# def xyz_for_layer(viewer_state, layer_state, mask=None):
#     xs = layer_state.layer[viewer_state.x_att][mask]
#     ys = layer_state.layer[viewer_state.y_att][mask]
#     zs = layer_state.layer[viewer_state.z_att][mask]
#     vals = [xs, ys, zs]
        
#     return array(list(zip(*vals)))


# # TODO: Make this better?
# # glue-plotly has had to deal with similar issues,
# # the utilities there are at least better than this
# def layer_color(layer_state):
#     layer_color = layer_state.color
#     if layer_color == '0.35' or layer_color == '0.75':
#         layer_color = 'gray'
#     return layer_color


# # For the 3D scatter viewer
# def scatter_layer_as_points(viewer_state, layer_state):
#     xyz = xyz_for_layer(viewer_state, layer_state)
#     return {
#         "mesh": xyz,
#         "color": "gray", # layer_color(layer_state),
#         "opacity": layer_state.alpha,
#         "style": "points_gaussian",
#         "point_size": 5 * layer_state.size,
#         "render_points_as_spheres": True
#     }


def xyz_bounds(viewer_state):
    return [(viewer_state.x_min, viewer_state.x_max),
            (viewer_state.y_min, viewer_state.y_max),
            (viewer_state.z_min, viewer_state.z_max)]


# Convert data to a format that can be used by the model-viewer
def process_data(data, layer_options, x_att="x", y_att="y", z_att="z", phi_resolution=15, theta_resolution=15) -> str:
    if data is not None:
        plotter = pv.Plotter()
        data_processed = []
        for i in range(data.shape[0]):
            data_processed.append([data[x_att][i], data[y_att][i], data[z_att][i]])
        pdata = pv.PolyData(data_processed)


        bounds = xyz_bounds(layer_options.viewer_state)
        factor = max((abs(b[1] - b[0]) for b in bounds))

        if layer_options.size_mode == "Fixed":
            radius = layer_options.size_scaling * sqrt((layer_options.size)) / (10 * factor)
            pc = pdata.glyph(geom=pv.Sphere(radius=radius,phi_resolution=phi_resolution, theta_resolution=theta_resolution), scale=False, orient=False)
        else:
            # The specific size calculation is taken from the scatter layer artist
            size_data = layer_options.layer[layer_options.size_att].ravel()
            size_data = clip(size_data, layer_options.size_vmin, layer_options.size_vmax)
            if layer_options.size_vmax == layer_options.size_vmin:
                sizes = sqrt(ones(size_data.shape) * 10)
            else:
                sizes = sqrt((20 * (size_data - layer_options.size_vmin) /
                        (layer_options.size_vmax - layer_options.size_vmin)))
            sizes *= (layer_options.size_scaling / factor)
            sizes[isnan(sizes)] = 0.
            pdata["sizes"] = sizes
            radius = layer_options.size_scaling * sqrt((layer_options.size)) / (10 * factor)
            pc = pdata.glyph(geom=pv.Sphere(radius=radius, phi_resolution=phi_resolution, theta_resolution=theta_resolution), scale="sizes", orient=False)


        fixed_color = layer_options is not None and (layer_options.cmap_mode == "Fixed" or layer_options.cmap is None)
        if fixed_color:
            color = layer_options.color
            plotter.add_mesh(pc, name="points", color=color, style="points_gaussian")
        else:
            points_per_sphere = 2 + (phi_resolution - 2) * theta_resolution
            cmap_values = layer_options.layer[layer_options.cmap_att]
            point_cmap_values = [y for x in cmap_values for y in (x,) * points_per_sphere]

            cmap = layer_options.cmap.name  # This assumes that we're using a matplotlib colormap
            clim = [layer_options.cmap_vmin, layer_options.cmap_vmax]
            if clim[0] > clim[1]:
                clim = [clim[1], clim[0]]
                cmap = f"{cmap}_r"

            pc.point_data["colors"] = point_cmap_values
            plotter.add_mesh(pc, scalars="colors", cmap=cmap, clim=clim, style="points_gaussian")

        path = os.getcwd() + f"/model_{data.label}.gltf"
        plotter.export_gltf(path)

        return str(path)
    else:
        return


class ModelViewer3DViewerState(ViewerState3D):
    
    def __init__(self, **kwargs):
        super(ModelViewer3DViewerState, self).__init__(**kwargs)

class ModelLayerStateWidget(v.VuetifyTemplate):
    template_file = (__file__, "modelviewer_options.vue")

    glue_state = GlueState().tag(sync=True)

    # Color

    cmap_mode_items = traitlets.List().tag(sync=True)
    cmap_mode_selected = traitlets.Int(allow_none=True).tag(sync=True)

    cmap_att_items = traitlets.List().tag(sync=True)
    cmap_att_selected = traitlets.Int(allow_none=True).tag(sync=True)

    cmap_items = traitlets.List().tag(sync=True)

    # Points

    points_mode_items = traitlets.List().tag(sync=True)
    points_mode_selected = traitlets.Int(allow_none=True).tag(sync=True)

    size_mode_items = traitlets.List().tag(sync=True)
    size_mode_selected = traitlets.Int(allow_none=True).tag(sync=True)

    size_att_items = traitlets.List().tag(sync=True)
    size_att_selected = traitlets.Int(allow_none=True).tag(sync=True)

    def __init__(self, layer_state):
        self.layer_state = layer_state
        self.glue_state = layer_state

        # Color

        link_glue_choices(self, layer_state, "cmap_mode")
        link_glue_choices(self, layer_state, "cmap_att")

        self.cmap_items = [
            {"text": cmap[0], "value": cmap[1].name} for cmap in colormaps.members
        ]

        # Points

        link_glue_choices(self, layer_state, "points_mode")
        link_glue_choices(self, layer_state, "size_mode")
        link_glue_choices(self, layer_state, "size_att")

        super(ModelLayerStateWidget, self).__init__()

    def vue_set_colormap(self, data):
        cmap = None
        for member in colormaps.members:
            if member[1].name == data:
                cmap = member[1]
                break
        self.layer_state.cmap = cmap


class ModelViewerStateWidget(widgets.VBox):
    def __init__(self, viewer_state):
        super(ModelViewerStateWidget, self).__init__()
        self.state = viewer_state


class ModelViewerLayerState(Scatter3DLayerState):
    def __init__(self, viewer_state=None, layer=None, **kwargs):
        super().__init__(viewer_state, layer, **kwargs)

class ModelViewerWidget(ipyreact.Widget):
    _esm = Path(__file__).parent / "modelviewer.mjs"
    model = traitlets.Any().tag(sync=True)
    viewer_height = traitlets.Any(default_value="400px").tag(sync=True)
    viewer = traitlets.Any()
    x_data = traitlets.Any()
    y_data = traitlets.Any()
    z_data = traitlets.Any()

    # @traitlets.observe('viewer')
    # def _on_viewer_change(self, change):
    #     self.update_view()

    # def __init__(self, viewer=None, viewer_height=None, x_data="x", y_data="y", z_data="z"):
    #     self.model.show()
    #     self.viewer = viewer
    #     self.x_data = x_data
    #     self.y_data = y_data
    #     self.z_data = z_data
    #     super(ModelViewerWidget, self).__init__()

    def update_view(self):
        if len(self.viewer.state.layers) != 0:
            layer_options = self.viewer.state.layers[0]
            data = self.viewer.state.layers[0].layer
        else:
            return
        self.model_path = process_data(data, layer_options, self.x_data, self.y_data, self.z_data)
        self.model = open(self.model_path, "rb").read()


class ModelViewerLayerArtist(LayerArtist):

    _layer_state_cls = ModelViewerLayerState

    def __init__(self, model_viewer, viewer_state, layer_state=None, layer=None):
        self._model_viewer = model_viewer
        super(ModelViewerLayerArtist, self).__init__(viewer_state, layer_state=layer_state, layer=layer)
        self._model_viewer.modelviewer_widget.data = layer.data

        on_change([(self.state, 'cmap_mode', 'cmap_att',
                    'cmap_vmin', 'cmap_vmax', 'cmap', 'color')])(self._refresh)
        on_change([(self.state, 'size', 'size_scaling',
                    'size_mode', 'size_vmin', 'size_vmax')])(self._refresh)

    
    def _refresh(self):
        self._model_viewer.redraw()

    def redraw(self):
        self._refresh()

    def update(self):
        self._refresh()

    def clear(self):
        self._refresh()

    def remove(self):
        data = None
        if self._model_viewer.layers:
            last_layer = self._model_viewer.layers[-1]
            data = last_layer.layer.data
        self._model_viewer.modelviewer_widget.data = data


@viewer_registry('model')
class ModelViewer(IPyWidgetView):
    # tools = ['model:ar']

    allow_duplicate_data = False
    allow_duplicate_subset = False

    _state_cls = ModelViewer3DViewerState
    _options_cls = ModelViewerStateWidget
    _data_artist_cls = ModelViewerLayerArtist
    _subset_artist_cls = ModelViewerLayerArtist
    _layer_style_widget_cls = ModelLayerStateWidget


    def __init__(self, session, x_data="x", y_data="y", z_data="z", state=None):
        super(ModelViewer, self).__init__(session, state=state)
        self.modelviewer_widget = ModelViewerWidget(viewer=self, viewer_height="500px", x_data="x", y_data="y", z_data="z")
        self.x_att = x_data
        self.y_att = y_data
        self.z_att = z_data

        self.create_layout()

    def redraw(self):
        # subsets = [k.layer for k in self.layers if isinstance(k.layer, Subset)]
        # with self.modelviewer_widget.hold_sync():
        #     self.modelviewer_widget.selections = [subset.label for subset in subsets]
        #     self.modelviewer_widget.selection_colors = [subset.style.color for subset in subsets]
        self.modelviewer_widget.update_view()

    @property
    def figure_widget(self):
        return self.modelviewer_widget