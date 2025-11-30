#pragma once

#include <stdexcept>
#include <string>

#ifdef _WIN32
// #define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <GL/gl.h>
#pragma comment(lib, "opengl32.lib")
#else
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>

#ifdef Success
#undef Success
#endif
#ifdef Bool
#undef Bool
#endif
#endif

namespace opengl_init {

    class OpenGLContext {
    public:
        OpenGLContext() {
#ifdef _WIN32
            init_win();
#else
            init_linux();
#endif
        }

        ~OpenGLContext() {
#ifdef _WIN32
            cleanup_win();
#else
            cleanup_linux();
#endif
        }

        OpenGLContext(const OpenGLContext&) = delete;
        OpenGLContext& operator=(const OpenGLContext&) = delete;

    private:
#ifdef _WIN32
        HWND hwnd_ = nullptr;
        HDC hdc_ = nullptr;
        HGLRC hrc_ = nullptr;

        void init_win() {
            WNDCLASSA wc = {};
            wc.lpfnWndProc = DefWindowProcA;
            wc.hInstance = GetModuleHandle(nullptr);
            wc.lpszClassName = "DummyGLWindow";

            if (!RegisterClassA(&wc))
                throw std::runtime_error("Failed to register window class");

            hwnd_ = CreateWindowExA(0, "DummyGLWindow", "Dummy", WS_OVERLAPPEDWINDOW, 0, 0, 1, 1, nullptr, nullptr,
                                    GetModuleHandle(nullptr), nullptr);
            if (!hwnd_)
                throw std::runtime_error("Failed to create dummy window");

            hdc_ = GetDC(hwnd_);
            if (!hdc_)
                throw std::runtime_error("Failed to get device context");

            PIXELFORMATDESCRIPTOR pfd = {};
            pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
            pfd.nVersion = 1;
            pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
            pfd.iPixelType = PFD_TYPE_RGBA;
            pfd.cColorBits = 32;
            pfd.cDepthBits = 24;
            pfd.cStencilBits = 8;
            pfd.iLayerType = PFD_MAIN_PLANE;

            int pixelFormat = ChoosePixelFormat(hdc_, &pfd);
            if (pixelFormat == 0)
                throw std::runtime_error("Failed to choose pixel format");

            if (!SetPixelFormat(hdc_, pixelFormat, &pfd))
                throw std::runtime_error("Failed to set pixel format");

            hrc_ = wglCreateContext(hdc_);
            if (!hrc_)
                throw std::runtime_error("Failed to create OpenGL context");

            if (!wglMakeCurrent(hdc_, hrc_))
                throw std::runtime_error("Failed to make OpenGL context current");
        }

        void cleanup_win() noexcept {
            if (hrc_) {
                wglMakeCurrent(nullptr, nullptr);
                wglDeleteContext(hrc_);
                hrc_ = nullptr;
            }
            if (hdc_) {
                ReleaseDC(hwnd_, hdc_);
                hdc_ = nullptr;
            }
            if (hwnd_) {
                DestroyWindow(hwnd_);
                hwnd_ = nullptr;
            }
        }

#else  // Linux / EGL
        EGLDisplay egl_display_ = EGL_NO_DISPLAY;
        EGLContext egl_context_ = EGL_NO_CONTEXT;
        EGLSurface egl_surface_ = EGL_NO_SURFACE;

        void init_linux() {
            auto eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
            auto eglGetPlatformDisplayEXT =
                    (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
            if (!eglQueryDevicesEXT || !eglGetPlatformDisplayEXT)
                throw std::runtime_error("EGL device extensions not available");

            EGLint num_devices = 0;
            if (!eglQueryDevicesEXT(0, nullptr, &num_devices) || num_devices == 0)
                throw std::runtime_error("No EGL devices found");

            EGLDeviceEXT* devices = new EGLDeviceEXT[num_devices];
            eglQueryDevicesEXT(num_devices, devices, &num_devices);

            egl_display_ = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[0], nullptr);
            delete[] devices;

            if (egl_display_ == EGL_NO_DISPLAY)
                throw std::runtime_error("Failed to get EGL display");

            EGLint major, minor;
            if (!eglInitialize(egl_display_, &major, &minor))
                throw std::runtime_error("Failed to initialize EGL");

            if (!eglBindAPI(EGL_OPENGL_API)) {
                eglTerminate(egl_display_);
                throw std::runtime_error("Failed to bind OpenGL API");
            }

            EGLint config_attribs[] = {EGL_SURFACE_TYPE,
                                       EGL_PBUFFER_BIT,
                                       EGL_RENDERABLE_TYPE,
                                       EGL_OPENGL_BIT,
                                       EGL_RED_SIZE,
                                       8,
                                       EGL_GREEN_SIZE,
                                       8,
                                       EGL_BLUE_SIZE,
                                       8,
                                       EGL_ALPHA_SIZE,
                                       8,
                                       EGL_DEPTH_SIZE,
                                       24,
                                       EGL_NONE};

            EGLConfig config;
            EGLint num_configs;
            if (!eglChooseConfig(egl_display_, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
                eglTerminate(egl_display_);
                throw std::runtime_error("Failed to choose EGL config");
            }

            EGLint pbuffer_attribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
            egl_surface_ = eglCreatePbufferSurface(egl_display_, config, pbuffer_attribs);
            if (egl_surface_ == EGL_NO_SURFACE) {
                eglTerminate(egl_display_);
                throw std::runtime_error("Failed to create EGL pbuffer surface");
            }

            EGLint context_attribs[] = {EGL_CONTEXT_MAJOR_VERSION,
                                        4,
                                        EGL_CONTEXT_MINOR_VERSION,
                                        5,
                                        EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                        EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
                                        EGL_NONE};

            egl_context_ = eglCreateContext(egl_display_, config, EGL_NO_CONTEXT, context_attribs);
            if (egl_context_ == EGL_NO_CONTEXT) {
                eglDestroySurface(egl_display_, egl_surface_);
                eglTerminate(egl_display_);
                throw std::runtime_error("Failed to create EGL context");
            }

            if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
                eglDestroyContext(egl_display_, egl_context_);
                eglDestroySurface(egl_display_, egl_surface_);
                eglTerminate(egl_display_);
                throw std::runtime_error("Failed to make EGL context current");
            }
        }

        void cleanup_linux() noexcept {
            if (egl_display_ != EGL_NO_DISPLAY) {
                eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
                if (egl_context_ != EGL_NO_CONTEXT)
                    eglDestroyContext(egl_display_, egl_context_);
                if (egl_surface_ != EGL_NO_SURFACE)
                    eglDestroySurface(egl_display_, egl_surface_);
                eglTerminate(egl_display_);
                egl_display_ = EGL_NO_DISPLAY;
            }
        }
#endif
    };

}  // namespace opengl_init
