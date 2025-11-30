import bpy
import os
from pathlib import Path


class KanahebiHydraRenderEngine(bpy.types.HydraRenderEngine):
    bl_idname = 'HYDRA_KANAHEBI'
    bl_label = "Kanahebi"
    bl_info = "Kanahebi Hydra Renderer"
    bl_use_preview = True
    bl_use_gpu_context = False
    bl_delegate_id = 'HdKanahebiRendererPlugin'

    @classmethod
    def register(cls):
        import pxr.Plug

        addon_dir = Path(__file__).parent
        plugin_path = str(addon_dir / "hdKanahebi" / "resources")

        if os.path.exists(plugin_path):
            pxr.Plug.Registry().RegisterPlugins([plugin_path])
        else:
            print(f"Warning: Plugin path not found: {plugin_path}")

    def get_render_settings(self, engine_type):
        settings = bpy.context.scene.hydra_kanahebi.viewport if engine_type == 'VIEWPORT' else bpy.context.scene.hydra_kanahebi.final

        result = {
            'kanahebi:global:targetsamples': settings.target_samples,
            "kanahebi:global:depth": settings.depth,
            'kanahebi:global:filmtransparent': bpy.context.scene.render.film_transparent,
            'kanahebi:global:exposure': bpy.context.scene.hydra_kanahebi.exposure,
        }
        if engine_type != 'VIEWPORT':
            result |= {
                'aovToken:Combined': "color",
            }
        return result


register, unregister = bpy.utils.register_classes_factory((
    KanahebiHydraRenderEngine,
))
