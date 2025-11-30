import bpy


class Properties(bpy.types.PropertyGroup):
    type = None

    @classmethod
    def register(cls):
        cls.type.hydra_kanahebi = bpy.props.PointerProperty(
            name="Hydra Kanahebi",
            description="Hydra Kanahebi properties",
            type=cls,
        )

    @classmethod
    def unregister(cls):
        del cls.type.hydra_kanahebi


class RenderProperties(bpy.types.PropertyGroup):
    target_samples: bpy.props.IntProperty(
        name="Target Samples",
        description="Number of samples per pixel",
        default=64,
        min=1,
        max=4096,
    )

    depth: bpy.props.IntProperty(
        name="Max Depth",
        description="Maximum ray bounce depth",
        default=16,
        min=1,
        max=64,
    )


class SceneProperties(Properties):
    type = bpy.types.Scene

    final: bpy.props.PointerProperty(type=RenderProperties)
    viewport: bpy.props.PointerProperty(type=RenderProperties)

    exposure: bpy.props.FloatProperty(
        name="Exposure",
        description="Exposure value",
        default=1.0,
        min=0.0,
        max=100.0,
    )


register, unregister = bpy.utils.register_classes_factory((
    RenderProperties,
    SceneProperties,
))
