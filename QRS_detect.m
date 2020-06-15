%author：JonneryR
%函数名定义

function [QRS_amp,QRS_ind] = QRS_detect(ecg_i,gr )

%保证输入的格式正确
if nargin < 2
    gr = 1; 
    if nargin<1
           error('The algorithm need a input:ecg_i.');
    end
end
if ~isvector(ecg_i)
  error('ecg_i must be a row or column vector.');
end
fs=360;
if size(ecg_i,2)<round(1.5*fs)+1
    error('The algorithm need a longer input.');
end
tic,
%s为输入信号
s=ecg_i;
N=size(s,2);
ECG=s;
FIR_c1=[0.0041,0.0053,0.0068,0.0080,0.0081,0.0058,-0.0000,-0.0097,-0.0226,...   
   -0.0370,-0.0498,-0.0577,-0.0576,-0.0477,-0.0278,0,0.0318,0.0625,0.0867,...    
    0.1000,0.1000,0.0867,0.0625,0.0318,0,-0.0278,-0.0477,-0.0576,-0.0577,...   
    -0.0498,-0.0370,-0.0226,-0.0097,-0.0000,0.0058,0.0081,0.0080,0.0068,...
    0.0053,0.0041];
FIR_c2=[0.0070,0.0094,0.0162,0.0269,0.0405,0.0555,0.0703,0.0833,0.0928,...    
    0.0979,0.0979,0.0928,0.0833,0.0703,0.0555,0.0405,0.0269,0.0162,0.0094,...    
    0.0070];

l1=size(FIR_c1,2);
ECG_l=[ones(1,l1)*ECG(1) ECG ones(1,l1)*ECG(N)]; 
ECG=filter(FIR_c1,1,ECG_l); 
ECG=ECG((l1+1):(N+l1)); 

a=round(0.015*fs);
b=round(0.060*fs);
Ns=N-2*b;          
S_l=zeros(1,b-a+1);
S_r=zeros(1,b-a+1);
S_dmax=zeros(1,Ns);
for i=1:Ns        
    for k=a:b
        S_l(k-a+1)=(ECG(i+b)-ECG(i+b-k))./k;
        S_r(k-a+1)=(ECG(i+b)-ECG(i+b+k))./k;
    end
  S_lmax=max(S_l);
  S_lmin=min(S_l);
  S_rmax=max(S_r);
  S_rmin=min(S_r);
  C1=S_rmax-S_lmin;
  C2=S_lmax-S_rmin;
  S_dmax(i)=max([C1 C2]);
end


l2=size(FIR_c2,2);
S_dmaxl=[ones(1,l2)*S_dmax(1) S_dmax ones(1,l2)*S_dmax(Ns)];
S_dmaxt=filter(FIR_c2,1,S_dmaxl);
S_dmaxt=S_dmaxt((l2+1):(Ns+l2));


w=8;wd=7;
d_l=[zeros(1,w) S_dmaxt zeros(1,w)]; 
m=zeros(1,Ns);
   for n=(w+1):(Ns+w)                
      m(n-w)=sum(d_l(n-w:n+w));       
   end
m_l=[ones(1,wd)*m(1) m ones(1,wd)*m(Ns)]; 


QRS_buf1=[];  
AMP_buf1=[];   
%这里是经验参数
thr_init0=0.4;thr_lim0=0.21;
thr_init1=0.6;thr_lim1=0.21; 
en=-1;        
thr0=thr_init0;
thr1=thr_init1;
thr1_buf=[]; 
thr0_buf=[];
for j=8:Ns
       t=1;
       cri=1;
       while t<=wd&&cri>0   
           cri=((m_l(j)-m_l(j-t))>0)&&(m_l(j)-m_l(j+t)>0);
           t=t+1;
       end
       if t==wd+1
           N1=size(QRS_buf1,2);               
           if m_l(j)>thr1                     
               if N1<2                        
                 QRS_buf1=[QRS_buf1 (j-wd)]; 
                 AMP_buf1=[AMP_buf1 m_l(j)];
                 en=1;
               else
                 dist=j-wd-QRS_buf1(N1);
                 if dist>0.24*fs               
                     QRS_buf1=[QRS_buf1 (j-wd)]; 
                     AMP_buf1=[AMP_buf1 m_l(j)];
                     en=1;
                 else
                     if m_l(j)>AMP_buf1(end)   
                         QRS_buf1(end)=j-wd;
                         AMP_buf1(end)=m_l(j);
                         en=1;
                     end     
                 end
               end
     
          else                                 
               
              if N1<2&&m_l(j)>thr0            
                  QRS_buf1=[QRS_buf1 (j-wd)];
                  AMP_buf1=[AMP_buf1 m_l(j)];
                  en=0;
              else
                if m_l(j)>thr0                 
                  dist_m=mean(diff(QRS_buf1));
                  dist=j-wd-QRS_buf1(N1);
                  if dist>0.24*fs && dist>0.5*dist_m  
                     QRS_buf1=[QRS_buf1 (j-wd)];
                     AMP_buf1=[AMP_buf1 m_l(j)];
                     en=0;
                  else
                      if m_l(j)>AMP_buf1(end)
                         QRS_buf1(end)=j-wd;
                         AMP_buf1(end)=m_l(j);
                         en=0;
                      end 
                  end
                else
                    en=-1;
                end
              end
           end
           N2=size(AMP_buf1,2);
           if N2>8
               AMP_buf1=AMP_buf1(2:9); 
           end
		  
           if en==1
            %对应公式中的参数
              thr1=0.65*mean(AMP_buf1);
              thr0=0.23*mean(AMP_buf1);
           else
               if en==0
                %对应公式中的参数
                   thr1=thr1-(abs(m_l(j)-mean(AMP_buf1)))/2;
                   thr0=0.35*m_l(j);
               end
           end
       end
       if thr1<=thr_lim1  
           thr1=thr_lim1;
       end
       
       if thr0<=thr_lim0
           thr0=thr_lim0;
       end
       
      thr1_buf=[thr1_buf thr1]; 
      thr0_buf=[thr0_buf thr0];
end
delay=round(l1/2)-2*w+2;
QRS_ind=QRS_buf1-delay;  
QRS_amp=s(QRS_ind);
toc
if gr==1    
   plot(m,'LineWidth',2);axis([1 size(m,2) -0.3 1.6*max(m)]);
   hold on;title('双斜率与自适应阈值__Jonnery/MEMS');grid on;
   plot(QRS_buf1,m(QRS_buf1),'ro','LineWidth',2);
   plot(thr1_buf,'r','LineWidth',2);
   plot(thr0_buf,'k','LineWidth',2);
   legend('Feature Signal','QRS Locations','Threshold1','Threshold0');
   plot(s,'LineWidth',2);%axis([1 size(s,2) min(s) 1.5*max(s)]);
   xlabel('n');ylabel('Voltage / mV');
   hold on;title('ECG识别QRS结果');grid on;
   plot(QRS_ind,QRS_amp,'ro','LineWidth',2);
   legend('Raw ECG','QRS Locations');
end
end

