from setuptools import setup, find_packages

setup(
    name="FastTrack",
    version="0.1.0",  
    author="Hanwen Kang",  
    author_email="kanghw@iphy.ac.cn", 
    description="we have demonstrated a fast, accurate, and flexible framework for predicting atomic diffusion barriers in crystalline solids by integrating universal machine‐learning force fields with three‐dimensional potential‐energy‐surface sampling and interpolation.",  
    url="https://github.com/atomly-materials-research-lab/FastTrack",  # 替换项目地址
    license="MIT",           # 根据LICENSE协议修改（GPL）
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pymatgen",
        "ase",
        "scipy",
        "matplotlib",
        "plotly",
        "pandas",
        "pymatgen-analysis-diffusion"
    ],
)

