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

local lda = function(x, y, n_comp)
    
    -- Make y a vector
    local y = y:clone():squeeze()
    local y_idx = torch.range(1,y:size(1)):long()
    
    local n_classes = y:max()

   -- Dependencies
   local decomposition = require 'decomposition'
    
    -- PCA to go to N-C dimensional space
    U = decomposition.pca(x, x:size(1)-n_classes, true)
    x = x * U
    
    -- Init
    local n = x:size(1)
    local d = x:size(2)
    
    -- Mean for each class
    local x_mean = torch.zeros(n_classes, d)
    local n_class = torch.zeros(n_classes)
    for c = 1,n_classes do
        local class_idx = y_idx[y:eq(c)]
        if class_idx:dim() > 0 then
            x_mean[c] = torch.mean(x:index(1, class_idx), 1)
            n_class[c] = class_idx:size(1)
        end
    end
        
    -- Calculate Si and Sw
    local Si = torch.Tensor(n_classes, d, d)
    local Sw = torch.Tensor(d, d):zero()
    
    for c = 1,n_classes do
        local class_idx = y_idx[y:eq(c)]
        if class_idx:dim() > 0 then
            local x_class = x:index(1, class_idx)
            local x_class_m = x_class - torch.ger(torch.ones(x_class:size(1)), x_mean[c]:squeeze())
        
            Si[c] = x_class_m:t() * x_class_m
            -- Si[c]:div(n_class[c] - 1)
        
            Sw:add(Si[c])
        end
    end
    
    -- Calculate S_b
    local mean = torch.mean(x, 1)
    local Sb = torch.Tensor(d, d):zero()
    for c = 1,n_classes do
        local mean_class_m = (x_mean[c] - mean):resize(1, d)
        local Sb_c = mean_class_m:t() * mean_class_m
        Sb_c:mul(n_class[c])
        Sb:add(Sb_c)
    end

    
    -- Calucalte Sw^{-1}Sb
    local Sw_inv = torch.inverse(Sw)
    local Sw_inv_Sb = Sw_inv * Sb
    
    -- Get eigenvalues and eigenvectors
    local ce, cv = torch.symeig(Sw_inv_Sb, 'V')
    -- Sort eigenvalues
    local ce, idx = torch.sort(ce, true)
    
    -- Sort eigenvectors
    cv = cv:index(2, idx)
    
    -- Keep C-1
    ce = ce:sub(1, n_classes-1)
    cv = cv:sub(1, -1, 1, n_classes-1)
    
    -- Total transform
    W = U * cv
    
    return W
end

return lda