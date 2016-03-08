function d = loadLabels(fileName)
  fid = fopen(fileName)
  bit32mask = [16^6 16^4 16^2 1];
  magicNumber = bit32mask * fread(fid, 4);
  numOfImages = bit32mask * fread(fid, 4);
  blockSize = 1;
  d = zeros(numOfImages, blockSize);
  for i = 1:numOfImages
    d(i, :) = fread(fid, blockSize);
  endfor;
  fclose(fid);
end