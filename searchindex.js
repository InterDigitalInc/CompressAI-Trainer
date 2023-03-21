Search.setIndex({docnames:["compressai_trainer/config","compressai_trainer/plot","compressai_trainer/registry","compressai_trainer/run","compressai_trainer/runners","compressai_trainer/typing","compressai_trainer/utils","index","tools/compressai","tools/eval_model","tools/rd_plotter","tools/train","tutorials/full","tutorials/installation"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["compressai_trainer/config.rst","compressai_trainer/plot.rst","compressai_trainer/registry.rst","compressai_trainer/run.rst","compressai_trainer/runners.rst","compressai_trainer/typing.rst","compressai_trainer/utils.rst","index.rst","tools/compressai.rst","tools/eval_model.rst","tools/rd_plotter.rst","tools/train.rst","tutorials/full.rst","tutorials/installation.rst"],objects:{"compressai_trainer.config":[[0,0,0,"-","config"],[0,0,0,"-","dataset"],[0,0,0,"-","engine"],[0,0,0,"-","env"],[0,0,0,"-","load"],[0,0,0,"-","outputs"]],"compressai_trainer.config.config":[[0,1,1,"","configure_conf"],[0,1,1,"","create_criterion"],[0,1,1,"","create_dataloaders"],[0,1,1,"","create_model"],[0,1,1,"","create_module"],[0,1,1,"","create_optimizer"],[0,1,1,"","create_scheduler"]],"compressai_trainer.config.dataset":[[0,2,1,"","DatasetTuple"],[0,1,1,"","create_data_transform"],[0,1,1,"","create_data_transform_composition"],[0,1,1,"","create_dataloader"],[0,1,1,"","create_dataset"],[0,1,1,"","create_dataset_tuple"]],"compressai_trainer.config.dataset.DatasetTuple":[[0,3,1,"","dataset"],[0,3,1,"","loader"],[0,3,1,"","transform"]],"compressai_trainer.config.engine":[[0,1,1,"","configure_engine"],[0,1,1,"","create_callback"],[0,1,1,"","create_logger"],[0,1,1,"","create_runner"]],"compressai_trainer.config.env":[[0,1,1,"","get_env"]],"compressai_trainer.config.load":[[0,1,1,"","get_checkpoint_path"],[0,1,1,"","load_checkpoint"],[0,1,1,"","load_config"],[0,1,1,"","state_dict_from_checkpoint"]],"compressai_trainer.config.outputs":[[0,1,1,"","write_config"],[0,1,1,"","write_git_diff"],[0,1,1,"","write_outputs"],[0,1,1,"","write_pip_list"],[0,1,1,"","write_pip_requirements"]],"compressai_trainer.plot":[[1,1,1,"","featuremap_image"],[1,1,1,"","featuremap_matplotlib"],[1,1,1,"","featuremap_matplotlib_looptiled"],[1,1,1,"","plot_rd"]],"compressai_trainer.registry":[[2,0,0,"-","catalyst"],[2,0,0,"-","torch"]],"compressai_trainer.registry.catalyst":[[2,1,1,"","register_callback"],[2,1,1,"","register_runner"]],"compressai_trainer.registry.torch":[[2,1,1,"","register_criterion"],[2,1,1,"","register_dataset"],[2,1,1,"","register_model"],[2,1,1,"","register_module"],[2,1,1,"","register_optimizer"],[2,1,1,"","register_scheduler"]],"compressai_trainer.run":[[3,0,0,"-","compressai"],[3,0,0,"-","eval_model"],[3,0,0,"-","plot_rd"],[3,0,0,"-","train"]],"compressai_trainer.run.eval_model":[[3,1,1,"","get_filenames"],[3,1,1,"","load_checkpoint_from_state_dict"],[3,1,1,"","load_model"],[3,1,1,"","main"],[3,1,1,"","run_eval_model"],[3,1,1,"","setup"],[3,1,1,"","write_results"]],"compressai_trainer.run.plot_rd":[[3,1,1,"","build_args"],[3,1,1,"","create_dataframe"],[3,1,1,"","main"],[3,1,1,"","plot_dataframe"]],"compressai_trainer.run.train":[[3,1,1,"","main"],[3,1,1,"","setup"]],"compressai_trainer.runners":[[4,2,1,"","BaseRunner"],[4,2,1,"","ImageCompressionRunner"]],"compressai_trainer.runners.BaseRunner":[[4,3,1,"","batch_meters"],[4,3,1,"","criterion"],[4,3,1,"","model"],[4,4,1,"","model_module"],[4,5,1,"","on_epoch_end"],[4,5,1,"","on_epoch_start"],[4,5,1,"","on_experiment_end"],[4,5,1,"","on_experiment_start"],[4,5,1,"","on_loader_end"],[4,5,1,"","on_loader_start"],[4,3,1,"","optimizer"]],"compressai_trainer.runners.ImageCompressionRunner":[[4,5,1,"","handle_batch"],[4,5,1,"","on_loader_end"],[4,5,1,"","on_loader_start"],[4,5,1,"","predict_batch"]],"compressai_trainer.typing":[[5,3,1,"","TCallback"],[5,3,1,"","TCriterion"],[5,3,1,"","TDataLoader"],[5,3,1,"","TDataset"],[5,3,1,"","TModel"],[5,3,1,"","TModule"],[5,3,1,"","TRunner"]],"compressai_trainer.utils":[[6,0,0,"-","git"],[6,0,0,"-","metrics"],[6,0,0,"-","pip"],[6,0,0,"-","system"],[6,0,0,"-","utils"]],"compressai_trainer.utils.aim":[[6,0,0,"-","query"]],"compressai_trainer.utils.aim.query":[[6,1,1,"","best_metric_index"],[6,1,1,"","get_runs_dataframe"],[6,1,1,"","metrics_at_index"],[6,1,1,"","pareto_optimal_dataframe"],[6,1,1,"","runs_by_query"]],"compressai_trainer.utils.catalyst":[[6,0,0,"-","loggers"]],"compressai_trainer.utils.catalyst.loggers":[[6,2,1,"","AimLogger"],[6,2,1,"","DistributionSuperlogger"],[6,2,1,"","FigureSuperlogger"]],"compressai_trainer.utils.catalyst.loggers.AimLogger":[[6,5,1,"","close_log"],[6,3,1,"","exclude"],[6,5,1,"","log_artifact"],[6,5,1,"","log_distribution"],[6,5,1,"","log_figure"],[6,5,1,"","log_hparams"],[6,5,1,"","log_image"],[6,5,1,"","log_metrics"],[6,4,1,"","logger"],[6,3,1,"","run"]],"compressai_trainer.utils.catalyst.loggers.DistributionSuperlogger":[[6,5,1,"","log_distribution"],[6,3,1,"","loggers"]],"compressai_trainer.utils.catalyst.loggers.FigureSuperlogger":[[6,5,1,"","log_figure"],[6,3,1,"","loggers"]],"compressai_trainer.utils.compressai":[[6,0,0,"-","results"]],"compressai_trainer.utils.compressai.results":[[6,1,1,"","compressai_dataframe"],[6,1,1,"","deep_codec_result"],[6,1,1,"","generic_codec_result"]],"compressai_trainer.utils.git":[[6,1,1,"","branch_name"],[6,1,1,"","commit_hash"],[6,1,1,"","common_ancestor_hash"],[6,1,1,"","diff"],[6,1,1,"","main_branch_name"]],"compressai_trainer.utils.metrics":[[6,1,1,"","compute_metrics"],[6,1,1,"","db"],[6,1,1,"","msssim"],[6,1,1,"","psnr"]],"compressai_trainer.utils.pip":[[6,1,1,"","freeze"],[6,1,1,"","list"]],"compressai_trainer.utils.system":[[6,1,1,"","hostname"],[6,1,1,"","username"]],"compressai_trainer.utils.utils":[[6,2,1,"","ConfigStringFormatter"],[6,1,1,"","arg_pareto_optimal_set"],[6,1,1,"","compute_padding"],[6,1,1,"","format_dataframe"],[6,1,1,"","np_img_to_tensor"],[6,1,1,"","num_parameters"],[6,1,1,"","tensor_to_np_img"]],"compressai_trainer.utils.utils.ConfigStringFormatter":[[6,5,1,"","get_field"]],compressai_trainer:[[0,0,0,"-","config"],[1,0,0,"-","plot"],[2,0,0,"-","registry"],[4,0,0,"-","runners"],[5,0,0,"-","typing"],[6,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","attribute","Python attribute"],"4":["py","property","Python property"],"5":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:attribute","4":"py:property","5":"py:method"},terms:{"0":[3,9,10,12],"02":10,"02d":12,"035":12,"1":[3,4,6,9,10,12],"128":12,"192":12,"1970":10,"2":[1,3,6,9,10],"24":12,"2d":1,"3":[3,10,13],"320":12,"3d":1,"4":6,"43800":12,"5":12,"5c6f152b":3,"6":[10,12],"8":[12,13],"boolean":6,"class":[0,2,4,6,12],"default":[6,10,12],"do":12,"export":12,"final":12,"float":[1,6],"function":[0,4,6,12],"import":[2,6,12],"int":[0,1,6],"new":[12,13],"public":1,"return":[0,1,4,6],"short":6,"true":[0,1,3,6,12],"try":1,A:0,By:12,For:[1,3,4,6,10,12],If:[6,10,12],In:12,It:12,One:[10,12],The:[3,4,9,12],Then:[12,13],These:7,To:[3,9,12,13],__init__:[2,12],_common_root:12,_debug_outputs_logg:4,_log_output:4,_lrschedul:0,_self_:12,abl:12,abov:[3,9,12],access:[2,12],account:12,accumul:6,activ:[6,12,13],ad:12,add:12,addit:[4,12],aforement:12,aggreg:[4,10],aim:[3,7,10],aimlogg:6,aimstack:6,alia:5,all:[2,4,10,12],allow:[2,12],also:[6,12],altern:[1,12],an:[2,3,4,6,12],ani:[0,1,3,4,6],annot:5,anyon:12,api:6,appli:1,appropri:12,ar:[4,6,12],arch:3,architectur:[3,8],area:1,arg:[3,4,6,10],arg_pareto_optimal_set:6,argument:1,argv:3,around:[3,8],arr:1,artifact:6,assist:7,authent:12,auto:12,automat:10,avail:[6,12],averag:[3,4,9],avoid:12,ax:1,b3d5bb2c5e3a6f49c69f39f6:10,b:6,base:[2,4,6,7,10,12],baserunn:[2,4,12],bashrc:12,basic:[4,6],batch:[3,4,6,12],batch_met:4,batch_siz:[3,9,12],befor:[4,12],begin:[4,12],below:12,best:[0,3,6,8],best_metric_index:6,better:12,between:1,bin:13,bind:12,bitstream:[3,9],bmshj2018:[3,8,10,12],bool:[0,1,6],both:10,bpp:[6,10],bpp_0:[6,10],bpp_1:[6,10],bpp_2:[6,10],bpp_3:10,branch:6,branch_nam:6,browser:12,build_arg:3,cach:3,call:[4,12],callabl:0,callback:[0,2,5],can:[3,10,12],cancel:12,candid:6,catalyst:[0,3,4,5,7,12],cbar:1,cd:[12,13],certain:12,channel:[1,4],checkpoint:[0,3,8,9],checkpoint_path:3,chosen:6,chw:1,ckpt:0,cli:3,clim:1,clone:13,close_log:6,cmap:1,code:[0,4,12,13],codec:6,codec_nam:6,collect:4,colorbar:1,colormap:1,column:[1,6],com:13,commit_hash:6,common:4,common_ancestor_hash:6,compar:12,complet:[3,11],compos:0,compress:[4,7,12],compressai:[2,4,9,12,13],compressai_datafram:6,compressai_train:[8,9,10,11,12],compressionmodel:[2,4,12],comput:[3,9],compute_metr:6,compute_pad:6,concret:2,conf:[0,3,12],config:[3,4,9],configstringformatt:6,configur:[0,2,3,7,9],configure_conf:0,configure_engin:0,connect:12,consid:12,consol:6,consolelogg:6,contain:[3,6,9,12],context:6,convert:6,copi:12,core:[0,3,5,6],correct:12,correspond:[6,12],cover:12,creat:[2,13],create_callback:0,create_criterion:0,create_data_transform:0,create_data_transform_composit:0,create_datafram:3,create_dataload:0,create_dataset:0,create_dataset_tupl:0,create_logg:0,create_model:0,create_modul:0,create_optim:0,create_runn:0,create_schedul:0,created_at:10,creation:[0,2],criterion:[2,4,10,12],csail:12,cuda:6,cuda_visible_devic:12,curl:13,current:[6,10,12],curv:[1,3,4,6,7],custom:[2,3,4,9],customimagecompressionrunn:[2,12],customrunn:6,d4e6e4c5e5d59c69f3bd7bd3:10,d:12,dar202:4,data:[0,3,4,5,6,8,9,10,12],datafram:[1,3,6],dataload:[0,4,5],dataparallel:4,dataset:[2,3,5,6,8,9,12],datasettupl:0,date:10,datetim:10,db:6,decor:2,deep_codec_result:6,def:[2,6,12],defin:[2,4],demonstr:12,depend:2,describ:[2,4],descript:3,desir:[10,12],destin:6,detail:12,determin:3,devic:[0,6,12],df:[1,3,6],dict:[0,1,3,4,6,12],dictconfig:[0,3],dictionari:4,diff:[4,6,12],differ:6,dimens:6,directli:12,directori:[3,6,9],displai:7,distribut:[4,6],distributeddataparallel:4,distributionsuperlogg:[4,6],divis:6,dl:[4,6],document:[4,6,8,9,10,11,12],done:12,download:12,due:1,dure:[3,4,7,9,12],dynam:[0,2],e4e6d4d5e5c59c69f3bd7be2:[0,3,8,9,10,12],e:[3,4,6,9,10,12],each:[4,6,10,12],easili:[3,8],echo:13,edit:13,edu:12,effect:[4,12],effortless:7,en:6,end:[6,7],engin:[7,12],enhanc:10,ensur:12,env:[12,13],environ:[0,4,12],epoch:[0,4,6,12],equival:[4,12],establish:12,etc:[2,3,4,6,12],eval:[3,9],eval_model:[8,9],evalu:3,event:4,exact:0,exampl:[0,3,4,6,8,12],example_eval_zoo:[3,9],example_experi:12,exclud:6,exist:12,exp:12,expect:12,experi:[3,4,6,7,10],factor:[3,8,10,12],factori:3,fallback:0,fals:[3,6,9,10],faster:1,featuremap:[1,4],featuremap_imag:1,featuremap_matplotlib:1,featuremap_matplotlib_looptil:1,few:12,field_nam:6,fig:6,fig_kw:1,figur:[1,6],figuresuperlogg:[4,6],file:[0,3,6,9,12],filenam:3,fill:12,fill_valu:1,find:12,firewal:12,first:[12,13],flag:[6,10],flatten:6,flexibl:7,follow:[3,4,9,12],format:[6,10],format_datafram:6,forward:12,frame:[3,6],framework:12,freez:6,fresh:12,from:[2,3,4,6,9,10,12],from_state_dict:3,frontier:6,full:[12,13],futur:[4,12],g:[3,4,6,10],gener:[3,4,12],generic_codec_result:6,get:[0,10],get_checkpoint_path:0,get_env:0,get_field:6,get_filenam:3,get_logg:6,get_runs_datafram:6,git:[4,12,13],github:13,give:7,given:[0,1,2,6,10],graphic:12,group:10,guid:[3,11,12],guidanc:4,ha:12,handl:4,handle_batch:[4,12],handler:4,hash:[4,6,10,12],head:6,height:6,help:13,here:12,histogram:4,home:[3,8,9],hostnam:6,how:12,hp:[3,10,12],hparam:[6,10],http:[6,12,13],hub:3,hwc:1,hydra:[7,12],hyperparamet:[0,12],i:[3,4,9,12],id:12,identifi:[3,9],ignor:[3,6,9],ilogg:[0,6],imag:[1,3,4,6,9,12],imagecompressionrunn:[4,12],imagefold:[3,9],imetr:4,implement:4,in_h:6,in_w:6,includ:[3,4,12],incom:12,index:6,infer:[3,4,6,9,12],infer_met:4,info:4,inform:[3,4,12],inherit:12,inner:4,input:[3,6],insid:12,instal:12,instanc:[4,12],instead:12,integr:7,interdigitalinc:13,intern:6,io:6,irunn:6,iter:6,its:[10,12],itself:12,json:[3,9],keep:6,keep_run_hash:6,kei:[6,10],keyword:1,kind:6,kodak:[3,6,8,12],kodim:12,kwarg:[4,6],l:12,lambda:6,lan:12,last:[0,3,6,9,12],latent:4,later:12,latest:6,layer:10,layout_kwarg:1,length:6,librari:7,limit:1,link:13,list:[4,6,10,12,13],liter:6,live:7,lmbda:[10,12],load:3,load_checkpoint:0,load_checkpoint_from_state_dict:3,load_config:0,load_model:3,loader:[0,3,4,6,9,12],local:[6,13],localhost:12,locat:12,log:[4,6,10,12],log_artifact:6,log_batch_metr:6,log_distribut:6,log_epoch_metr:6,log_figur:6,log_hparam:6,log_imag:6,log_metr:6,logger:[0,4,6],logger_typ:0,loop:[1,2,4],loss:[4,6],lower:1,lr_schedul:0,m:[2,3,8,9,10,12,13],machin:12,made:[4,12],mai:[3,9,10,12],main:[3,6,10,12],main_branch_nam:6,make:[2,4],manag:[7,13],map:[0,2,4],master:6,match:6,matplotlib:1,max:6,meta:[3,9],meter:4,method:[2,3,4],metric:[1,3,4,7,9,10,12],metrics_at_index:6,min:6,min_div:6,min_metr:6,minim:6,misc:12,miscellan:6,mit:12,mkdir:12,model:[0,2,3,4,6,10],model_checkpoint:[3,9,12],model_modul:4,model_nam:6,modifi:12,modul:[0,2,3,5,6],monitor:6,more:[3,4,7,8,12],move:12,ms:6,mse:[3,6],msssim:6,multi:10,multipl:[6,10],must:12,my_custom_model:[2,12],mycustommodel:[2,12],n:[2,12],name:[2,3,6,9,10,12],nan:6,nc:12,ncol:1,ndarrai:[1,6],nest:1,net:[0,6],net_aux:12,network:7,neural:7,nn:[0,3,5,6],non:0,none:[1,6],noqa:4,note:6,notimplementederror:4,now:12,np_img_to_tensor:6,nrow:1,num:4,num_epoch:[4,12],num_fil:3,num_paramet:6,num_sampl:[3,9],num_work:[3,9],number:1,numpi:[1,6],object:[0,2,6],omegaconf:[0,3],on_batch_end:[4,12],on_batch_start:[4,12],on_epoch_end:[4,12],on_epoch_start:[4,12],on_experiment_end:[4,12],on_experiment_start:[4,12],on_loader_end:[4,12],on_loader_start:[4,12],onc:[4,12],one:[3,10,12],onli:[6,10],onto:6,op:6,open:12,opt_metr:6,optim:[0,2,4,6,10,12],option:[1,6],order:[1,4,12],ordereddict:0,org:13,organ:12,origin:12,other:[6,12],our:12,out_h:6,out_w:6,output:[3,4,6,9],output_dir:[3,9],outsid:12,over:12,overrid:[3,9],own:4,p:12,packag:[0,13],pad:[1,6],panda:[3,6],param:4,paramet:[0,1,4,6,12],pareto:6,pareto_optimal_datafram:6,part:6,particular:[0,6],pass:[1,4],patch:12,path:[0,3,8,9,13],path_to_artifact:6,paus:12,pd:1,per:[3,4,9],pin:13,pip:[4,13],plasma:1,platform:7,pleas:[3,4,11,12],plot:[3,4,6],plot_datafram:3,plot_rd:[1,10],plotter:[3,10],png:12,poetri:12,point:[6,10],port:12,pre:[4,12],predic:6,predict_batch:4,prefix:10,prepar:6,pretrain:3,previous:12,primari:6,printf:12,prior:3,process:4,produc:10,properti:[4,6],provid:[4,12],pseudo:[4,12],psnr:[1,6,10],psnr_0:[6,10],psnr_1:[6,10],psnr_2:[6,10],psnr_3:10,psnr_base:10,psnr_enhanc:10,psnr_low:10,psnr_rgb:[6,10],psnr_yuv:[6,10],pth:[3,8,9,12],py:[2,4,12],python3:13,python:[3,6,8,9,10,12,13],qualiti:[1,3,12],queri:[3,6],quickli:12,r0k:12,rais:4,rang:[4,12],rare:[4,12],rate:[4,10],ratedistortionloss:12,rd:[1,3,4,7],readthedoc:6,reassembl:0,recommend:12,reconfigur:12,reconstruct:[3,9],reducelronplateau:[0,12],regist:[2,12],register_callback:2,register_criterion:2,register_dataset:2,register_model:[2,12],register_modul:2,register_optim:2,register_runn:[2,12],register_schedul:2,registri:12,reimplement:4,releas:[4,12],relev:[10,12],remain:1,remote_serv:12,repo:[3,6,10],repositori:13,reproduc:[7,12],requir:[3,12,13],research:7,respect:[10,12],restrict:12,result:[3,6,9,12],rev1:6,rev2:6,rev:6,rgb:[6,10],root:[0,3,6,9,12],row:1,run:[0,4,6,8,9,10,11,13],run_eval_model:3,run_hash:[6,12],run_root:0,runnabl:3,runner:[0,2,3,5,6,8,9],runs_by_queri:6,runs_root:12,runtim:[2,12],s:[3,4,12],same:[0,6,10],sampl:[3,9],save:[3,4,9,12],scale:6,scatter_kwarg:1,schedul:[2,12],scope:6,script:[3,8],sdk:6,section:12,see:[3,4,11,12],selector:10,self:[2,6,12],separ:10,seri:6,set:[1,3,6,9,12],setup:3,shape:4,shell:[12,13],should:[2,6,12],show:[1,10],shown:12,shuffl:[3,9],similarli:12,simpl:10,sinc:10,singl:[6,10],size:12,skip:6,skip_nan:6,slightli:1,slow:1,some:[10,12],some_custom_argu:12,somewher:12,sourc:[0,1,2,3,4,6,9,12,13],specif:[10,12],specifi:[4,6,10],split:[3,9,12],src:12,ssh:12,ssim:6,ssl:13,standard:3,start:12,startswith:10,state:[0,12],state_dict:12,state_dict_from_checkpoint:0,step:[4,6],str:[0,1,2,3,4,6],string:2,suffix:6,supervisedrunn:6,symlink:12,system:0,tag:6,tar:3,task:4,tcallback:5,tcriterion:5,tdataload:[0,5],tdataset:[0,5],tell:12,templat:12,tensor:[0,1,6],tensor_to_np_img:6,tensorboard:12,test:[3,4,8,9,12],test_exp:6,text:6,them:2,thi:[0,1,2,6,10,12],those:12,tile:1,titl:1,tmodel:5,tmodul:5,tmp_run:12,tofu:12,tool:7,torch:[0,3,5,6],torchcriterion:4,torchoptim:4,torchvis:0,totensor:[3,9],track:12,tracker:[3,7,10,12],train:[0,2,4,6,7,8,9],train_check:12,trainer:[3,4,9,12,13],transform:[0,3,9],transform_conf:0,tri:0,trunner:5,tsv:[3,9],tupl:[1,3,6],type:[3,9,12],ui:12,union:[0,6],uniqu:[10,12],unit:6,unnecessari:12,unpad:6,unus:6,unzip:12,up:12,updat:4,upgrad:13,url:12,us:[2,3,4,6,9,10],usag:[3,8,12],user:[3,10,12],usernam:[6,12],util:[0,3,4,5],v1:6,vae:10,valid:[4,6,12],valu:[1,6,10,12],variabl:12,variou:[3,12],verifi:0,version:13,via:[2,10],videocompressionrunn:[4,12],vimeo90k:12,vimeo_triplet:12,virtual:12,visibl:12,visual:[7,12],wa:12,wai:12,walkthrough:[3,11],warn_onli:0,we:[4,12],web:12,weight:4,wget:12,what:10,where:[10,12],whether:1,which:[3,4,6,9],width:6,wise:4,within:[10,12],without:12,work:[3,8,12],wrapper:[3,8],write_config:0,write_git_diff:0,write_output:0,write_pip_list:0,write_pip_requir:0,write_result:3,written:12,x:[6,10],x_hat:6,x_object:6,xs:6,y:[6,10],y_object:6,yaml:[2,4,7],yet:4,your:4,yuv:[6,10],zip:12,zoo:[3,9]},titles:["compressai_trainer.config","compressai_trainer.plot","compressai_trainer.registry","compressai_trainer.run","compressai_trainer.runners","compressai_trainer.typing","compressai_trainer.utils","CompressAI Trainer","Use <code class=\"docutils literal notranslate\"><span class=\"pre\">compressai.utils</span></code>","Evaluate a model","Plot an RD curve","Train a model","Walkthrough","Installation"],titleterms:{"public":12,aim:[6,12],aim_repo:10,an:10,argument:12,basic:12,catalyst:[2,6],check:12,checkpoint:12,cli:[10,12],command:12,compressai:[3,6,7,8],compressai_train:[0,1,2,3,4,5,6],config:[0,12],configur:12,continu:12,creat:12,curv:10,custom:12,dashboard:12,dataset:0,defin:12,directori:12,engin:0,env:0,environ:13,eval_model:3,evalu:9,exampl:10,experi:12,git:6,gpu:12,help:10,host:12,instal:13,line:12,load:[0,12],local:12,loop:12,metric:6,model:[9,11,12],multi:12,navig:12,onli:12,output:[0,12],overrid:12,own:12,pareto:10,path:12,pip:6,plot:[1,10],plot_rd:3,poetri:13,previou:12,privat:12,queri:10,quick:12,rd:10,registri:2,remot:12,repositori:12,resum:12,run:[3,12],runner:[4,12],saniti:12,singl:12,specifi:12,structur:12,system:6,tip:12,torch:2,train:[3,11,12],trainer:7,type:5,us:[8,12,13],util:[6,8],venv:13,via:12,view:12,virtual:13,walkthrough:12,weight:12,yaml:12,your:12}})