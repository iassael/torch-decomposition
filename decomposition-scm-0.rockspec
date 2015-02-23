package = "decomposition"
version = "scm-0"

source = {
  url = "git://github.com/iassael/torch7-decomposition",
  dir = "decomposition"
}

description = {
  summary = "Decomposition module for Torch7.",
  detailed = [[
    - Principal Component Analysis (PCA)
    - Whitened Principal Component Analysis (W-PCA)
    - Linear Discriminant Analysis (LDA)
    - Locality Preserving Projections (LPP)
    - Neighbourhood Preserving Projections (NPP)
    - Fast Independent Component Analysis (FastICA)
  ]],
  homepage = "https://github.com/iassael/torch7-decomposition",
  license = "MIT"
}

dependencies = {
  "torch >= 7.0",
}

build = {
  type = "builtin",
  modules = {
    decomposition = "./decomposition.lua"
  }
}