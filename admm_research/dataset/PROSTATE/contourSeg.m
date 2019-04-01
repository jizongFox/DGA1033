function [SegCont] = contourSeg(ImageSrc,Seg,Color,Thickness)

[Cont,Thick] = extractContours(Seg,Thickness);
SegCont = ImageSrc;
Red = ImageSrc;
Green = ImageSrc;
Blue = ImageSrc;
Idx = find(Thick); 
Red(Idx) = Color(1);
Green(Idx) = Color(2);
Blue(Idx) = Color(3);
SegCont(:,:,1) = Red;
SegCont(:,:,2) = Green;
SegCont(:,:,3) = Blue;

