# JPEG-Image-Compressor
A Python program that compresses raw images based on the JPEG compression algorithm.

This program takes as input a raw image (eg: .bmp).

The image is read using the OpenCV library in BGR color space, then converted to YCrCb. Each channel is normalized by subtracting 128. Then a 4: 2: 2 subsampling scheme is applied (another scheme can be used), by utilizing a 2 × 2 averaging filter on the chrominance channels (another type of filter can be used), thus reducing the number of bits per pixel to 8 + 4 + 4 = 16.

Each channel is divided into 8 × 8 blocks – and is padded with zeros if needed. Each block undergoes a discrete cosine transform, where in the resulting block, the first component of each block is called the DC coefficient, and the other 63 are AC components.

DC coefficients are encoded using DPCM as follows: \<size in bits\>, \<amplitude\>. AC components are encoded using run length in the following way: \<run length, size in bits\>, \<amplitude\>, while using zigzag scan on the block to produce longer runs of zeros.
  
An intermediary stream consists of encoded DC and AC components, and an EOB (end of block) to mark the end of the block. To achieve a higher compression rate, all zero AC components are trimmed from the end of the zigzag scan.
  
A Huffman dictionary is created by calculating the frequency of each intermediary symbol. Since one image is to be sent in this project, the frequencies of the intermediary symbols will be calculated from those of this image (one can use a predefined Huffman dictionary). Each intermediary stream is encoded using its assigned codeword. The encoded bitstream is then written to an output file.

# Note
Kindly check the poll in the discussions tab.

If anyone codes the decoder, kindly let me know so that I can link your repository from here.

Thanks
