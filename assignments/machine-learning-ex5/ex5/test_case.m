X = [1 2 ; 1 3 ; 1 4 ; 1 5];
y = [7 6 5 4]';
Xval = [1 7 ; 1 -2];
yval = [2 12]'
[lambda_vec_actual, error_train_actual, error_val_actual] = validationCurve(X,y,Xval,yval);

lambda_vec_expected = [0.00000 0.00100 0.00300 0.01000 0.03000 0.10000 0.30000 1.00000 3.00000 10.00000]'
error_train_expected = [0.00000 0.00000 0.00000 0.00000 0.00002 0.00024 0.00200 0.01736 0.08789 0.27778]'
error_val_expected = [0.25000 0.25055 0.25165 0.25553 0.26678 0.30801 0.43970 1.00347 2.77539 6.80556]'

if (lambda_vec_actual == lambda_vec_expected) 
  disp('lambda is OK');
endif


if (error_train_actual == error_train_expected) 
  disp('error_train is OK'); else printf("error train actual: "); printf("%.5f ", error_train_actual); printf("\nerror train expected: ");printf("%.5f ", error_train_expected);
endif