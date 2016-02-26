%% Calculates F1 score based on number of true/false positive/negative examples.
%% tp - true positive
%% tn - true negative
%% fp - false positive
%% fn - false negative
function F1 = calculateF1score(tp, tn, fp, fn)
  tmp1 = tp + fp;
  tmp2 = tp + fn;
  if ((tmp1 != 0) && (tmp2 != 0))
    Prec = tp/tmp1;
    Rec = tp/tmp2;
  else 
    Prec = 0;
    Rec = 0;
  endif;
  % avoid division-by-zero error
  if (Prec * Rec == 0) 
    F1 = 0;
  else 
    F1 = 2*Prec*Rec/(Prec + Rec);
  endif;
end
