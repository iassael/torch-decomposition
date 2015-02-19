--[=====[

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

--]=====]

function npp(x, n_comp)
    
    -- Size
    local N = x:size(1)
    
    -- Heat kernel temperature parameter
    local t = 1000
    
    -- Epsilon threshold
    local eps = 10000
    
    -- Build euclidean distances matrix
    local dist = torch.zeros(N, N);
    for i = 1,N do
        for j = 1,N do
            -- Compute norm distance
            local dist_i_j = torch.norm(x[{{i}}] - x[{{j}}])
            
            -- If distance < epsilon then not a neighbour
            if dist_i_j < eps then
                dist[{{i},{j}}] = dist_i_j
            else
                dist[{{i},{j}}] = 0
            end
        end
    end
    
    -- Build W
    local W = torch.exp(-dist / t);
    
    -- Build D, a diagonal matrix continaing the sum of distances of Wi
    local D = torch.diag(torch.sum(W, 2):squeeze())
    
    -- Build Laplacian Matrix
    local L = D - W
    
    -- NPP
    local M = (torch.eye(W:size(1)) - W) * (torch.eye(W:size(1)) - W):t()
    local C = x:t() * x
    local L = x:t() * M * x
    local W_eig = torch.inverse(C) * L
    
    -- LPP
    --  local W_eig = torch.inverse(x:t() * D * x) * x:t() * L * x
    
    -- Solve generalized eigenvector problem to obtain solutions
    local ce, cv = torch.symeig(W_eig, 'V')

    -- Sort eigenvalues
    local ce, idx = torch.sort(ce, true)
    local cv = cv:index(2, idx)
    
    -- Remove small eigenvalues
    local mask = ce:lt(1e-03)
    local mask_idx = torch.range(1,ce:size(1)):long()
    ce = ce:index(1, mask_idx[mask])
    cv = cv:index(2, mask_idx[mask])
    
    -- Keep n_comp
    if n_comp and n_comp < cv:size(2) then
        ce = ce:sub(1, n_comp)
        cv = cv:sub(1, -1, 1, n_comp)
    end
    
    local A = cv
    
    return A
end