Xvalue_max=0;
Yvalue_max=0;
Xvalue_min=0;
Yvalue_min=0;
for i=1:50
   len=length(database{i,3});
   % Find max and min
   max_sublen=max(cellfun(@length,database{i,3}));
   for j=1:len
       P=database{i,3}{j};
       if Xvalue_max<max(P(1:2:end))
           Xvalue_max = max(P(1:2:end));
       end
       if Yvalue_max<max(P(2:2:end))
           Yvalue_max = max(P(2:2:end));
       end
      if Xvalue_min>min(P(1:2:end))
           Xvalue_min = min(P(1:2:end));
       end
       if Yvalue_min>min(P(2:2:end))
           Yvalue_min = min(P(2:2:end));
       end
   end
end
%%%%%%%%%
%% Transformations
%%%%%%%%%
size_data=50;
Xmean_value=(Xvalue_max+Xvalue_min)/size_data;
Ymean_value=(Yvalue_max+Yvalue_min)/size_data;
idx=1;
for i=100:150
   subplot(8,8,idx);
   len=length(database{i,3});
   max_sublen=max(cellfun(@length,database{i,3}));
   P=[];
   hold on;
   for j=1:len
       P=database{i,3}{j};
       %P(1:2:end)=(P(1:2:end)- Xmean_value)/(Xvalue_max+Xvalue_min);
       %P(2:2:end)=(P(2:2:end)- Ymean_value)/(Yvalue_max+Yvalue_min);
       P(1:2:end)=(P(1:2:end)+Xmean_value);
       P(2:2:end)=(P(2:2:end)+Ymean_value);
       %P=P*mean_value;
       plot(P(1:2:end),P(2:2:end));
   end
   title(database{i,1});
   idx=idx+1;
   axis([Xvalue_min Xvalue_max Yvalue_min Yvalue_max]);
   hold off;
end
