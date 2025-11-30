#pragma once

enum class SampleType {
    Reflection = 1,
    Transmission = 2,
    GlossyReflection = 3,
    GlossyTransmission = 4,
    DiffuseReflection = 5,
    DiffuseTransmission = 6,
    SpecularReflection = 7,
    SpecularTransmission = 8,
};

enum class ScatterMode {
    RT,
    R,
    T,
};
