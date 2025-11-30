FROM aswf/ci-base:2025.3

# ===========================
# 開発ツール
# ===========================
RUN dnf install -y epel-release && \
  dnf install -y cmake && \
  dnf clean all

# ===========================
# dnfプラグインのインストール
# ===========================
RUN dnf install -y 'dnf-command(config-manager)' && dnf clean all

# ===========================
# CUDA 13.0 のインストール
# ===========================
# CUDAリポジトリの追加
RUN dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# CUDA Toolkit 13.0 のインストール
RUN dnf install -y cuda-toolkit-13-0 && dnf clean all

# CUDA環境変数
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# ===========================
# OptiX 9 SDKのインストール
# ===========================
# OptiXインストーラーをコンテナにコピー
COPY ./NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh /tmp/

# OptiXのインストール
RUN mkdir -p /opt/optix && \
  chmod +x /tmp/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh && \
  /tmp/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh --skip-license --prefix=/opt/optix && \
  rm /tmp/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh

# OptiX環境変数
ENV OptiX_INSTALL_DIR=/opt/optix
ENV OptiX_ROOT=/opt/optix
ENV PATH=${OptiX_INSTALL_DIR}/bin:${PATH}

# ===========================
# GCC Toolset 11 の有効化
# ===========================
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-11/root/usr/lib64:$LD_LIBRARY_PATH
ENV MANPATH=/opt/rh/gcc-toolset-11/root/usr/share/man:$MANPATH

# ===========================
# ユーザーの作成
# ===========================
ARG USERNAME=aswf
ARG USER_UID=${LOCAL_UID:-1000}
ARG USER_GID=${LOCAL_GID:-1000}

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /workspace
CMD ["/bin/bash"]
