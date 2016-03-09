function d = loadLabels(fileName, n)
  fid = fopen(fileName)
  bit32mask = [16^6 16^4 16^2 1];
  magicNumber = bit32mask * fread(fid, 4);
  numOfImages = bit32mask * fread(fid, 4);
  blockSize = 1;
  imagesToRead = min(n, numOfImages);
  d = zeros(imagesToRead, blockSize);
  for i = 1:imagesToRead
    d(i, :) = fread(fid, blockSize);
  endfor;
  fclose(fid);
end