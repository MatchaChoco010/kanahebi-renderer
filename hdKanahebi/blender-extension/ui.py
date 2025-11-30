import bpy
from .engine import KanahebiHydraRenderEngine


class Panel(bpy.types.Panel):
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    COMPAT_ENGINES = {KanahebiHydraRenderEngine.bl_idname}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES


#
# Final render settings
#
class KANAHEBI_HYDRA_RENDER_PT_final(Panel):
    """Final render settings"""
    bl_idname = 'KANAHEBI_HYDRA_RENDER_PT_final'
    bl_label = "Final Render Settings"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        settings = context.scene.hydra_kanahebi.final

        layout.prop(settings, 'target_samples')
        layout.prop(settings, 'depth')


#
# Viewport render settings
#
class KANAHEBI_HYDRA_RENDER_PT_viewport(Panel):
    """Viewport render settings"""
    bl_idname = 'KANAHEBI_HYDRA_RENDER_PT_viewport'
    bl_label = "Viewport Render Settings"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        settings = context.scene.hydra_kanahebi.viewport

        layout.prop(settings, 'target_samples')
        layout.prop(settings, 'depth')


class KANAHEBI_HYDRA_RENDER_PT_film(Panel):
    bl_label = "Film"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(context.scene.render, "film_transparent", text="Transparent Background")
        layout.prop(context.scene.hydra_kanahebi, "exposure")


register_classes, unregister_classes = bpy.utils.register_classes_factory((
    KANAHEBI_HYDRA_RENDER_PT_final,
    KANAHEBI_HYDRA_RENDER_PT_viewport,
    KANAHEBI_HYDRA_RENDER_PT_film,
))


def register():
    register_classes()


def unregister():
    unregister_classes()
