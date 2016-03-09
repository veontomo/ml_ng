function d = loadData(fileName, n)
  fid = fopen(fileName)
  bit32mask = [16^6 16^4 16^2 1];
  magicNumber = bit32mask * fread(fid, 4);
  numOfImages = bit32mask * fread(fid, 4)
  rows = bit32mask * fread(fid, 4);
  cols = bit32mask * fread(fid, 4);
  blockSize = rows*cols;
  imagesToRead = min(n, numOfImages);
  d = zeros(imagesToRead, blockSize);
  for i = 1:imagesToRead
    d(i, :) = fread(fid, blockSize);
  endfor;
  fclose(fid);
end