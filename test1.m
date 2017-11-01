addpath jsonlab/
data=loadjson('kaggle/iceberg/train.json', 'ShowProgress', 1);
all_sum_1 = 0;
all_length_1 = 0;
max_1 = -100;
min_1 = 100;
for i=1:5
  V1 = data(1,i){1:1}.band_1;
  all_sum_1 += sum(V1);
  all_length_1 += length(V1);
  max_1 = max([max_1, V1]);
  min_1 = min([min_1, V1]);
endfor;

all_u_1 = all_sum_1 / all_length_1;

for i=1:5
  V1 = data(1,i){1:1}.band_1;
  M1 = reshape(V1, 75, 75);
  u1 = sum(V1)/length(V1);
  nM1 = (M1 .- all_u_1)./(max_1 - min_1);

  V2 = data(1,i){1:1}.band_2;
  M2 = reshape(V2, 75, 75);
  u2 = sum(V2)/length(V2);
  nM2 = (M2 .- u2)./(max(V2) - min(V2));

  subplot(5, 4, ((i-1) * 4)+1, 'align');imagesc(M1);
  title(data(1,i){1:1}.is_iceberg);
  axis('off');
  subplot(5, 4, ((i-1) * 4)+2, 'align');imagesc(nM1);
  title(data(1,i){1:1}.is_iceberg);
  axis('off');
  subplot(5, 4, ((i-1) * 4)+3, 'align');imagesc(M2);
  title(data(1,i){1:1}.is_iceberg);
  axis('off');
  subplot(5, 4, ((i-1) * 4)+4, 'align');imagesc(nM2);
  title(data(1,i){1:1}.is_iceberg);
  axis('off');
endfor;