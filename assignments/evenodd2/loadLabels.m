function d = loadLabels(fileName, n)
  fid = fopen(fileName)
  bit32mask = [16^6 16^4 16^2 1];
  magicNumber = bit32mask * fread(fid, 4);
  numOfImages = bit32mask * fread(fid, 4);
  blockSize = 1;
  imagesToRead = min(n, numOfImages);
  d = zeros(imagesToRead, 10);
  for i = 1:imagesToRead
    pos = fread(fid, 1);
    if pos == 0 
      pos = 10 
    endif;
    d(i, pos) = 1;
  endfor;
  fclose(fid);
end