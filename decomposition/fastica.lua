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

local fastica = function(x, y, n_comp)
    -- Functions and Derivatives
    local g = torch.tanh
    local g_prime = function(u)
        return torch.ones(u:size()) - torch.tanh(u):pow(2)
    end

   -- Dependencies
   local decomposition = require 'decomposition'
    
    -- Init sizes
    M = x:size(1)
    N = x:size(2)
    C = M - 1
    
    -- Center Data
    local mean = torch.mean(x, 1)
    local x = x - torch.ger(torch.ones(x:size(1)), mean:squeeze())    
    
    -- Whiten data
    U = decomposition.pca(x, C, true)
    x = x * U
    
    -- Transpose for same notation
    x = x:t()
    
    -- Init W and wp with random variables
    local W = torch.rand(C, C)
    
    -- Iterate
    local max_iter = 100
    for p = 1, C do
        local converged = false
        local iter = 1
        
        -- While Wp changes
        while not converged and iter <= max_iter do
            local Wp_old = W[{{p}}]:clone()
            
            -- Compute new Wp
            local Wp_x = Wp_old * x
            local Wp = torch.ones(1, C) * (x * g(Wp_x):t()):mean() - Wp_old * g_prime(Wp_x):mean(2):squeeze()
            
            -- Normalise it
            Wp = Wp / torch.norm(Wp)
            W[{{p}}] = Wp
            
            local sum_w = torch.zeros(1, C);
            if p > 1 then
                --  for j = 1, p-1 do
                --     sum_w = sum_w + W[{{j}}] * W[{{j}}]:t() * Wp
                --  end
                sum_w = ((Wp * W[{{1,p-1}}]:t()) * W[{{1,p-1}}])
            end
            Wp = Wp - sum_w     
            Wp = Wp / torch.norm(Wp)
                      
            
            -- Check convergence
            local lim = 1 - torch.abs(Wp * W[{{p}}]:t()):squeeze()
            local converged = lim < 1e-04
            iter = iter + 1
            
            if iter > 1 then
                W[{{p}}] = Wp
            end
        end
    end
    
    -- De-whitening
    ICA = U * W
    
    -- Keep n_comp
    if n_comp and n_comp < cv:size(2) then
        ICA = ICA:sub(1, -1, 1, n_comp)
    end
    
    return ICA
end

return fastica