# Decomposition module for Torch7

- Principal Component Analysis (PCA)

- Whitened Principal Component Analysis (W-PCA)

- Linear Discriminant Analysis (LDA)

- Locality Preserving Projections (LPP)

- Neighbourhood Preserving Projections (NPP)

- Fast Independent Component Analysis (FastICA)

by John-Alexander Assael

http://www.johnassael.com

https://github.com/iassael/component_analysis_torch7

## Installation
You can clone this repository or download the module using:

 `luarocks install decomposition`.

## Usage

Call `decomposition = require "decomposition"`
and then any of the following:

- `decomposition.pca(x)`,

- `decomposition.lda(x, y)`,

- `decomposition.lpp(x)`,

- `decomposition.npp(x)`,

- `decomposition.fastica(x)`.


Alternativly, you can use iTorch notebook and open `decomposition.ipynb`.

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request

## Notes

The implementations were developed in terms of learning and may not be optimal.

## License

Copyright (C) 2015 John-Alexander Assael (www.johnassael.com)
https://github.com/iassael/component_analysis_torch7

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.