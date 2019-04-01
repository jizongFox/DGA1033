function [Cont, Thick] = extractContours(Seg,Thickness)

Cont = zeros(size(Seg));
Cont = bwperim(Seg == 1, 8);

Thick = bwmorph(Cont, 'thicken', Thickness);
