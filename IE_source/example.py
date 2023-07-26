def Full_experiment_AttentionalIE_PDE_Navier_Stokes(model, Encoder, Decoder, Data, time_seq, index_np, mask, times, args, extrapolation_points): # experiment_name, plot_freq=1):
    # scaling_factor=1
    
    
    #metadata for saving checkpoints
    if args.model=='nie': 
        str_model_name = "nie"
    elif args.model=='node': 
        str_model_name = "node"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         # -- logger location
        #writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        #print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        #with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
        #    for key, value in args.__dict__.items(): 
        #        f.write('%s:%s\n' % (key, value))



    obs = Data
    times = time_seq
    
    
    if Encoder is None and Decoder is None:
        All_parameters = model.parameters()
    elif Encoder is not None and Decoder is None:
        All_parameters = list(model.parameters())+list(Encoder.parameters())
    elif Decoder is not None and Encoder is None:
        All_parameters = list(model.parameters())+list(Decoder.parameters())
    else:
        All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0,last_epoch=-1)# Emanuele's version
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1) #My first version
    #scheduler = LRScheduler(optimizer,patience = 20,min_lr=1e-12,factor=0.1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0,last_epoch=-1)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        if Encoder is None or Decoder is None:
            model, optimizer, scheduler, pos_enc, pos_dec, f_func = load_checkpoint(path, model, optimizer, scheduler, None, None,  None)
        else:
            G_NN, optimizer, scheduler, model, Encoder, Decoder = load_checkpoint(path, None, optimizer, scheduler, model, Encoder, Decoder)

    
    if args.eqn_type=='Navier-Stokes':
        spatial_domain_xy = torch.meshgrid([torch.linspace(0,1,args.n_points) for i in range(2)])
        
        x_space = spatial_domain_xy[0].flatten().unsqueeze(-1)
        y_space = spatial_domain_xy[1].flatten().unsqueeze(-1)
        
        spatial_domain = torch.cat([x_space,y_space],-1)
    
    
    if args.mode=='train':
        #lr_scheduler = LRScheduler(optimizer,patience = 50,min_lr=1e-5,factor=0.1)
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        # Loss_print = []
        # Val_Loss = []
        all_train_loss=[]
        all_val_loss=[]
        
            
        Data_splitting_indices = Train_val_split(np.copy(index_np),0)
        Train_Data_indices = Data_splitting_indices.train_IDs()
        Val_Data_indices = Data_splitting_indices.val_IDs()
        print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
        print('Train_Data_indices: ',Train_Data_indices)
        print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
        print('Val_Data_indices: ',Val_Data_indices)
        
        # Train Neural IDE
        get_times = Select_times_function(times,extrapolation_points)

        save_best_model = SaveBestModel()
        start = time.time()
        
        split_size = int(args.training_split*obs.size(0))
        
        if args.eqn_type == 'Burgers':
            obs_train = obs[:obs.size(0)-split_size,:,:]
        else:
            obs_train = obs[:obs.size(0)-split_size,:,:,:]
            
        for i in range(args.epochs):
            
            if args.support_tensors is True or args.support_test is True:
                if args.combine_points is True:
                    sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1))
                    temp_sampled_tensors = sampled_tensors
                    sampled_tensors = sampled_tensors.to(device)
                    #Check if there are duplicates and resample if there are
                    sampled_tensors = torch.cat([times,sampled_tensors])
                    dup=np.array([0])
                    while dup.size != 0:
                        u, c = np.unique(temp_sampled_tensors, return_counts=True)
                        dup = u[c > 1]
                        if dup.size != 0:
                            sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1))
                            sampled_tensors = sampled_tensors.to(device)
                            sampled_tensors = torch.cat([times,sampled_tensors])
                    dummy_times=sampled_tensors
                    real_idx=real_idx[:times.size(0)]
                if args.combine_points is False:
                        dummy_times = torch.linspace(times[0],times[-1],args.sampling_points)
            
            model.train()
            if Encoder is not None:
                Encoder.train()
            if Decoder is not None:
                Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            # GPUtil.showUtilization()
            counter=0
            train_loss = 0.0
            
            if args.n_batch>1:
                if args.eqn_type == 'Burgers':
                    obs_shuffle = obs_train[torch.randperm(obs_train.size(0)),:,:]
                else:
                    obs_shuffle = obs_train[torch.randperm(obs_train.size(0)),:,:,:]
                
            for j in tqdm(range(0,obs.size(0)-split_size,args.n_batch)):
                
                if args.n_batch==1:
                    if args.eqn_type == 'Burgers':
                        Dataset_train = Dynamics_Dataset(obs_train[j,:,:],times)
                    else:
                        Dataset_train = Dynamics_Dataset(obs_train[j,:,:,:],times)
                else:
                    if args.eqn_type == 'Burgers':
                        Dataset_train = Dynamics_Dataset(obs_shuffle[j:j+args.n_batch,:,:],times,args.n_batch)
                    else:
                        Dataset_train = Dynamics_Dataset(obs_shuffle[j:j+args.n_batch,:,:,:],times,args.n_batch)
                #Dataset_val = Dynamics_Dataset(obs[j-split_size,:,:],times)
                # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
                # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

                # For the sampler
                train_sampler = SubsetRandomSampler(Train_Data_indices)
                #valid_sampler = SubsetRandomSampler(Val_Data_indices)

                # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

                dataloaders = {'train': torch.utils.data.DataLoader(Dataset_train, sampler=train_sampler,\
                                                                    batch_size = args.n_batch, drop_last=True),
                              }

                train_loader = dataloaders['train']
                #val_loader = dataloaders['val']
                #loader_test = dataloaders['test']

            #for obs_, ts_, ids_ in tqdm(train_loader): 
                obs_, ts_, ids_ = Dataset_train.__getitem__(index_np)#next(iter(train_loader))
                
                obs_ = obs_.to(args.device)
                ts_ = ts_.to(args.device)
                ids_ = torch.from_numpy(ids_).to(args.device)
                # obs_, ts_, ids_ = next(iter(loader))

                ids_, indices = torch.sort(ids_)
                ts_ = ts_[indices]
                ts_ = torch.cat([times[:1],ts_])
                if args.n_batch==1:
                    if args.eqn_type == 'Burgers':
                        obs_ = obs_[indices,:]
                    else:
                        if Encoder is None:
                            obs_ = obs_[indices,:,:]
                            obs_ = obs_[:,indices,:]
                else:
                    if args.eqn_type == 'Burgers':
                        obs_ = obs_[:,indices,:]
                    else:
                        if Encoder is None:
                            obs_ = obs_[:,indices,:,:]
                            obs_ = obs_[:,:,indices,:]
                            
                    
                if args.perturbation_to_obs0 is not None:
                       perturb = torch.normal(mean=torch.zeros(obs_.shape[1]).to(args.device),
                                              std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                else:
                    perturb = torch.zeros_like(obs_[0]).to(args.device)
                # print('obs_[:5]: ',obs_[:5])
                # print('ids_[:5]: ',ids_[:5])
                # print('ts_[:5]: ',ts_[:5])

                # print('obs_: ',obs_)
                # print('ids_: ',ids_)
                # print('ts_: ',ts_)

                # obs_, ts_ = obs_.squeeze(1), ts_.squeeze(1)
                if args.initialization is False:
                    y_init = None
                    
                    if args.n_batch==1:
                        if args.eqn_type == 'Burgers':
                            c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_[:,:1])
                            interpolation = NaturalCubicSpline(c_coeffs)
                            c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                        else:
                            c= lambda x: obs_[:,:,:1].repeat(1,1,args.time_points).unsqueeze(-1).to(device)
                    else:
                        if args.eqn_type == 'Burgers':
                            c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_[:,:,:1])
                            interpolation = NaturalCubicSpline(c_coeffs)
                            c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,1,args.time_points).unsqueeze(-1)
                        else:
    #                         c= lambda x: \
    #                         Encoder(obs_[:,:,:,:1].repeat(1,1,1,args.time_points)\
    #                         .permute(0,3,1,2).requires_grad_(True)).unsqueeze(-1)\
    #                         .permute(0,2,3,1,4).contiguous().to(args.device)
                            c= lambda x: \
                            Encoder(obs_[:,:,:,:1].permute(0,3,1,2).requires_grad_(True))\
                                    .permute(0,2,3,1).unsqueeze(-2)\
                                    .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
        
                else:
                    y_init = Encoder(obs_[:,:,:,:1].permute(0,3,1,2).requires_grad_(True))\
                                    .permute(0,2,3,1).unsqueeze(-2)\
                                    .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                    c = lambda x: torch.zeros_like(y_init).to(args.device)
                    
                
                #if args.patches is True:
                if args.eqn_type == 'Navier-Stokes':
#                     y_0 = Encoder(obs_[:,:,:,:1].repeat(1,1,1,args.time_points)\
#                         .permute(0,3,1,2)).unsqueeze(-1)\
#                         .permute(0,2,3,1,4)[:,:,:,:1,:]
                    y_0 =  Encoder(obs_[:,:,:,:1].permute(0,3,1,2))\
                                .permute(0,2,3,1).unsqueeze(-2)
                    
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                if args.support_tensors is False:
                    if args.n_batch==1:
                        if args.eqn_type == 'Burgers':
                            z_ = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                        else:
                            z_ = Integral_spatial_attention_solver(
                                    times_integration.to(args.device),
                                    obs_[:,:,0].unsqueeze(-1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                    else:
                        if args.eqn_type == 'Burgers':
                            z_ = Integral_spatial_attention_solver_multbatch(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_[:,0].unsqueeze(-1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                        else:
                            z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(args.device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=args.initialization
                                    ).solve()
                else:
                    z_ = Integral_spatial_attention_solver(
                            torch.linspace(0,1,args.time_points).to(device),
                            obs_[0].unsqueeze(0).to(args.device),
                            c=c,
                            sampling_points = args.time_points,
                            support_tensors=dummy_times.to(device),
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            spatial_integration=True,
                            spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                            spatial_domain_dim=1,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            smoothing_factor=args.smoothing_factor,
                            output_support_tensors=True
                            ).solve()
                    if args.combine_points is True:
                        z_ = z_[real_idx,:]
                
                
                if args.eqn_type=='Burgers':
                    if args.n_batch==1:
                            z_ = z_.view(args.n_points,args.time_points)
                            z_ = torch.cat([z_[:,:1],z_[:,-1:]],-1)
                    else:
                            z_ = z_.view(args.n_batch,args.n_points,args.time_points)
                            z_ = torch.cat([z_[:,:,:1],z_[:,:,-1:]],-1)
                else:
                    if args.n_batch==1:
                        z_ = z_.view(args.n_points,args.n_points,args.time_points)
                    else:
                        z_ = z_.view(z_.shape[0],args.n_points,args.n_points,args.time_points,args.dim)
                    
                    if Decoder is not None:
#                         z_ = z_.squeeze(-1).permute(0,3,1,2)
#                         z_ = Decoder(z_.requires_grad_(True)).permute(0,2,3,1)
                        z_ = Decoder(z_.requires_grad_(True))
                    else:
                        z_ = z_.view(args.n_batch,Data.shape[1],Data.shape[2],args.time_points)
                    if args.initial_t is False:
                        obs_ = obs_[:,:,:,1:]
                     
                #loss_ts_ = get_times.select_times(ts_)[1]
                loss = F.mse_loss(z_, obs_.detach()) #Original 
                # print('z_[:,:].to(args.device): ',z_[:,:].to(args.device))
                # print('obs_.to(args.device).detach()[:,:]: ',obs_.to(args.device).detach()[:,:])
                # loss = F.mse_loss(z_[:,:].to(args.device), obs_.to(args.device).detach()[:,:]) #Original 

                
                # ###############################
                # Loss_print.append(to_np(loss))
                # ###############################

                optimizer.zero_grad()
                loss.backward()#(retain_graph=True)
                optimizer.step()

                # n_iter += 1
                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  split_size==0 and args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, ts_, z_, ids_

            ## Validating
                
            model.eval()
            if Encoder is not None:
                Encoder.eval()
            if Decoder is not None:
                Decoder.eval()
                
            with torch.no_grad():

                    #Only do this if there is a validation dataset
                
                val_loss = 0.0
                counter = 0
                if split_size>0:
                    # for images, _, _, _, _ in tqdm(val_loader):   # frames, timevals, angular_velocity, mass_height, mass_xpos
                    for j in tqdm(range(obs.size(0)-split_size,obs.size(0),args.n_batch)):
                        
                        valid_sampler = SubsetRandomSampler(Train_Data_indices)
                        if args.n_batch==1:
                            if args.eqn_type == 'Burgers':
                                Dataset_val = Dynamics_Dataset(obs[j,:,:],times)
                            else:
                                Dataset_val = Dynamics_Dataset(obs[j,:,:,:],times)
                        else:
                            if args.eqn_type == 'Burgers':
                                Dataset_val = Dynamics_Dataset(obs[j:j+args.n_batch,:,:],times,args.n_batch)
                            else:
                                Dataset_val = Dynamics_Dataset(obs[j:j+args.n_batch,:,:,:],times,args.n_batch)
                        
                        val_loader = torch.utils.data.DataLoader(Dataset_val, sampler=valid_sampler,\
                                                                 batch_size = args.n_batch, drop_last=True)
                    
                    #for obs_val, ts_val, ids_val in tqdm(val_loader):
                        obs_val, ts_val, ids_val = Dataset_val.__getitem__(index_np)#next(iter(val_loader))
                        obs_val = obs_val.to(args.device)
                        ts_val = ts_val.to(args.device)
                        
                        ids_val = torch.from_numpy(ids_val).to(args.device)

                        ids_val, indices = torch.sort(ids_val)
                        # print('indices: ',indices)
                        if args.n_batch ==1:
                            if args.eqn_type == 'Burgers':
                                obs_val = obs_val[indices,:]
                            else:
                                if Encoder is None:
                                    obs_val = obs_val[indices,:,:]
                                    obs_val = obs_val[:,indices,:]
                        else:
                            if args.eqn_type == 'Burgers':
                                obs_val = obs_val[:,indices,:]
                            else:
                                if Encoder is None:
                                    obs_val = obs_val[:,indices,:,:]
                                    obs_val = obs_val[:,:,indices,:]
                        
                        ts_val = ts_val[indices]
                                             

                        #Concatenate the first point of the train minibatch
                        # obs_[0],ts_
                        # print('\n In validation mode...')
                        # print('obs_[:5]: ',obs_[:5])
                        # print('ids_[:5]: ',ids_[:5])
                        # print('ts_[:5]: ',ts_[:5])
                        # print('ts_[0]:',ts_[0])

                        ## Below is to add initial data point to val
                        #obs_val = torch.cat((obs_[0][None,:],obs_val))
                        #ts_val = torch.hstack((ts_[0],ts_val))
                        #ids_val = torch.hstack((ids_[0],ids_val))

                        # obs_val, ts_val, ids_val = next(iter(loader_val))
                        # print('obs_val.shape: ',obs_val.shape)
                        # print('ids_val: ',ids_val)
                        # print('ts_val: ',ts_val)

                        # obs_val, ts_val = obs_val.squeeze(1), ts_val.squeeze(1)
                        if args.initialization is False:
                            y_init=None
                            if args.n_batch==1:
                                if args.eqn_type == 'Burgers':
                                    c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_val[:,:1])
                                    interpolation = NaturalCubicSpline(c_coeffs)
                                    c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                                else:
                                    c= lambda x: obs_val[:,:,:1].repeat(1,1,args.time_points).unsqueeze(-1).to(device)
                            else:
                                if args.eqn_type == 'Burgers':
                                    c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_val[:,:,:1])
                                    interpolation = NaturalCubicSpline(c_coeffs)
                                    c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,1,args.time_points).unsqueeze(-1)
                                else:
    #                                 c= lambda x: \
    #                                             Encoder(obs_val[:,:,:,:1].repeat(1,1,1,args.time_points)\
    #                                             .permute(0,3,1,2)).unsqueeze(-1)\
    #                                             .permute(0,2,3,1,4).contiguous().to(args.device)
                                    c= lambda x: \
                                        Encoder(obs_val[:,:,:,:1].permute(0,3,1,2))\
                                                .permute(0,2,3,1).unsqueeze(-2)\
                                                .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                        else:
                            y_init = Encoder(obs_val[:,:,:,:1].permute(0,3,1,2))\
                                                .permute(0,2,3,1).unsqueeze(-2)\
                                                .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                            c = lambda x: torch.zeros_like(y_init).to(args.device)
                        
                        if args.eqn_type == 'Navier-Stokes':
#                             y_0 = Encoder(obs_val[:,:,:,:1].repeat(1,1,1,args.time_points)\
#                             .permute(0,3,1,2)).unsqueeze(-1)\
#                             .permute(0,2,3,1,4)[:,:,:,:1,:]
                            y_0 = Encoder(obs_val[:,:,:,:1].permute(0,3,1,2))\
                                            .permute(0,2,3,1).unsqueeze(-2)\
                                            .to(args.device)
                            
                            
                        if args.ts_integration is not None:
                            times_integration = args.ts_integration
                        else:
                            times_integration = torch.linspace(0,1,args.time_points)
                    
                        if args.support_tensors is False:
                            if args.n_batch==1:
                                if args.eqn_type == 'Burgers':
                                    z_val = Integral_spatial_attention_solver(
                                            torch.linspace(0,1,args.time_points).to(device),
                                            obs_val[0].unsqueeze(1).to(args.device),
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                            spatial_domain_dim=1,
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            ).solve()
                                else:
                                    z_val = Integral_spatial_attention_solver(
                                            torch.linspace(0,1,args.time_points).to(device),
                                            obs_val[:,:,0].unsqueeze(-1).to(args.device),
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= spatial_domain.to(device),
                                            spatial_domain_dim=2,
                                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            ).solve()
                                    
                            else:
                                if args.eqn_type == 'Burgers':
                                    z_val = Integral_spatial_attention_solver_multbatch(
                                        torch.linspace(0,1,args.time_points).to(device),
                                        obs_val[:,0].unsqueeze(-1).to(args.device),
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                        spatial_domain_dim=1,
                                        #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                        #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        ).solve()
                                else:
                                    z_val = Integral_spatial_attention_solver_multbatch(
                                            times_integration.to(args.device),
                                            y_0.to(args.device),
                                            y_init=y_init,
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= spatial_domain.to(args.device),
                                            spatial_domain_dim=2,
                                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            initialization=args.initialization
                                            ).solve()
                                
                        else:
                            z_val = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    support_tensors=dummy_times.to(device),
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    output_support_tensors=True
                                    ).solve()
                        
                            if args.combine_points is True:
                                z_val = z_val[real_idx,:]
                          
                        if args.eqn_type=='Burgers':
                            if args.n_batch==1:
                                    z_val = z_val.view(args.n_points,args.time_points)
                                    z_val = torch.cat([z_val[:,:1],z_val[:,-1:]],-1)
                            else:
                                    z_val = z_val.view(args.n_batch,args.n_points,args.time_points)
                                    z_val = torch.cat([z_val[:,:,:1],z_val[:,:,-1:]],-1)
                        else:
                            if args.n_batch==1:
                                z_val = z_val.view(args.n_points,args.n_points,args.time_points)
                            else:
                                z_val = z_val.view(z_val.shape[0],args.n_points,args.n_points,args.time_points,args.dim)
                            
                            if Decoder is not None:
#                                 z_val = z_val.squeeze(-1).permute(0,3,1,2)
#                                 z_val = Decoder(z_val).permute(0,2,3,1)
                                z_val = Decoder(z_val)
                            else:
                                z_val = z_val.view(args.n_batch,Data.shape[1],Data.shape[2],args.time_points)
                            if args.initial_t is False:
                                obs_val = obs_val[:,:,:,1:]
                            
                        #validation_ts_ = get_times.select_times(ts_val)[1]
                        loss_validation = F.mse_loss(z_val, obs_val.detach())
                        # Val_Loss.append(to_np(loss_validation))
                        
                        del obs_val, ts_val, z_val, ids_val

                        counter += 1
                        val_loss += loss_validation.item()
                        
                        del loss_validation

                        #LRScheduler(loss_validation)
                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)
                
                
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            #writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
            #if len(all_val_loss)>0:
            #    writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
            #if args.lr_scheduler == 'ReduceLROnPlateau':
            #    writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
            #elif args.lr_scheduler == 'CosineAnnealingLR':
            #    writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)

            
            with torch.no_grad():
                
                model.eval()
                if Encoder is not None:
                    Encoder.eval()
                if Decoder is not None:
                    Decoder.eval()
                
                if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
                    # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
                        
                    plt.plot(np.log10(all_train_loss),label='Train loss')
                    if split_size>0:
                        plt.plot(np.log10(all_val_loss),label='Val loss')
                    plt.xlabel("Epoch")
                    plt.ylabel("MSE Loss")
                    # timestr = time.strftime("%Y%m%d-%H%M%S")
                    #plt.show()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))

                    #for j in tqdm(range(0,obs.size(0),args.n_batch)):
                    for j in tqdm(range(1)):
                        if args.n_batch==1:
                            if args.eqn_type == 'Burgers':
                                Dataset_all = Dynamics_Dataset(Data[j,:,:],times)
                            else:
                                Dataset_all = Dynamics_Dataset(Data[j,:,:,:],times)
                        else:
                            if args.eqn_type == 'Burgers':
                                Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:],times,args.n_batch)
                            else:
                                Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:,:],times,args.n_batch)
                                
                        loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = args.n_batch)

                        obs_test, ts_test, ids_test = Dataset_all.__getitem__(index_np)#next(iter(loader_test))

                        ids_test, indices = torch.sort(torch.from_numpy(ids_test))
                        # print('indices: ',indices)
                        if args.n_batch==1:
                            if args.eqn_type == 'Burgers':
                                obs_test = obs_test[indices,:]
                            else:
                                if Encoder is None:
                                    obs_test = obs_test[indices,:,:]
                                    obs_test = obs_test[:,indices,:]
                        else:
                            if args.eqn_type == 'Burgers':
                                obs_test = obs_test[:,indices,:]
                            else:
                                if Encoder is None:
                                    obs_test = obs_test[:,indices,:,:]
                                    obs_test = obs_test[:,:,indices,:]
                        ts_test = ts_test[indices]
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)


                        obs_test = obs_test.to(args.device)
                        ts_test = ts_test.to(args.device)
                        ids_test = ids_test.to(args.device)
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)
                        # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                        if args.initialization is False:
                            y_init=None
                            if args.n_batch ==1:
                                if args.eqn_type == 'Burgers':
                                    c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:1])
                                    interpolation = NaturalCubicSpline(c_coeffs)
                                    c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                                else:
                                    c = lambda x: obs_test[:,:,:1].repeat(1,1,args.time_points).unsqueeze(-1).to(device)
                            else:
                                if args.eqn_type == 'Burgers':
                                    c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:,:1])
                                    interpolation = NaturalCubicSpline(c_coeffs)
                                    c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,1,args.time_points).unsqueeze(-1)
                                else:
    #                                 c= lambda x: \
    #                                             Encoder(obs_test[:,:,:,:1].repeat(1,1,1,args.time_points)\
    #                                             .permute(0,3,1,2)).unsqueeze(-1)\
    #                                             .permute(0,2,3,1,4).contiguous().to(args.device)
                                    c= lambda x: Encoder(obs_test[:,:,:,:1].permute(0,3,1,2))\
                                                .permute(0,2,3,1).unsqueeze(-2)\
                                                .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                        else:
                            y_init = Encoder(obs_test[:,:,:,:1].permute(0,3,1,2))\
                                                .permute(0,2,3,1).unsqueeze(-2)\
                                                .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                            c = lambda x: torch.zeros_like(y_init).to(args.device)
                        if args.eqn_type == 'Navier-Stokes':
#                             y_0 = Encoder(obs_test[:,:,:,:1].repeat(1,1,1,args.time_points)\
#                             .permute(0,3,1,2)).unsqueeze(-1)\
#                             .permute(0,2,3,1,4)[:,:,:,:1,:]
                            y_0 = Encoder(obs_test[:,:,:,:1].permute(0,3,1,2))\
                                            .permute(0,2,3,1).unsqueeze(-2)\
                                            .to(args.device)
                            
                        if args.ts_integration is not None:
                            times_integration = args.ts_integration
                        else:
                            times_integration = torch.linspace(0,1,args.time_points)
                                  
                        if args.support_test is False:
                            if args.n_batch==1:
                                if args.eqn_type == 'Burgers':
                                    z_test = Integral_spatial_attention_solver(
                                            torch.linspace(0,1,args.time_points).to(device),
                                            obs_test[0].unsqueeze(1).to(args.device),
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                            spatial_domain_dim=1,
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            ).solve()
                                else:
                                    z_test = Integral_spatial_attention_solver(
                                            torch.linspace(0,1,args.time_points).to(device),
                                            obs_test[:,:,0].unsqueeze(-1).to(args.device),
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= spatial_domain.to(device),
                                            spatial_domain_dim=2,
                                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            ).solve()
                                    
                            else:
                                if args.eqn_type == 'Burgers':
                                    z_test = Integral_spatial_attention_solver_multbatch(
                                        torch.linspace(0,1,args.time_points).to(device),
                                        obs_test[:,0].unsqueeze(-1).to(args.device),
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                        spatial_domain_dim=1,
                                        #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                        #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        ).solve()
                                else:
                                    z_test = Integral_spatial_attention_solver_multbatch(
                                            times_integration.to(args.device),
                                            y_0.to(args.device),
                                            y_init=y_init,
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= spatial_domain.to(args.device),
                                            spatial_domain_dim=2,
                                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            initialization=args.initialization
                                            ).solve()
                        else:
                            z_test = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    support_tensors=dummy_times.to(device),
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    output_support_tensors=True,
                                    ).solve()
                            
                        #print('Parameters are:',ide_trained.parameters)
                        #print(list(All_parameters))
                        
                        
                        if args.eqn_type == 'Burgers':
                            if args.n_batch== 1:
                                z_test = z_test.view(args.n_points,args.time_points)
                                z_test = torch.cat([z_test[:,:1],z_test[:,-1:]],-1)
                            else:
                                z_test = z_test.view(args.n_batch,args.n_points,args.time_points)
                                z_test = torch.cat([z_test[:,:,:1],z_test[:,:,-1:]],-1)
                            new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                            plt.figure(1,facecolor='w')

                            z_p = z_test
                            if args.n_batch >1:
                                z_p = z_test[0,:,:]
                            z_p = to_np(z_p)

                            if args.n_batch >1:
                                obs_print = to_np(obs_test[0,:,:])
                            else:
                                obs_print = to_np(obs_test[:,:])

#                             if obs.size(2)>2:
#                                 z_p = pca_proj.fit_transform(z_p)
#                                 obs_print = pca_proj.fit_transform(obs_print)                    

                            plt.figure(1, facecolor='w')
                            plt.plot(torch.linspace(0,1,args.n_points),z_p[:,0],c='r', label='model')
                            plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,0],c='r',s=10)


                            # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                            plt.scatter(torch.linspace(0,1,obs_test.size(1)),obs_print[:,0],label='Data',c='blue', alpha=0.5)
                            #plt.xlabel("dim 0")
                            #plt.ylabel("dim 1")
                            #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                            plt.legend()
                            # plt.show()
                            # timestr = time.strftime("%Y%m%d-%H%M%S")
                            plt.savefig(os.path.join(path_to_save_plots,'plot_t0_epoch'+str(i)+'_'+str(j)))

                            plt.figure(2, facecolor='w')

                            plt.plot(torch.linspace(0,1,args.n_points),z_p[:,1], label='model')
                            plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,1],s=10)


                            # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                            plt.scatter(torch.linspace(0,1,obs_test.size(1)),obs_print[:,1],label='Data',c='blue', alpha=0.5)
                            #plt.xlabel("dim 0")
                            #plt.ylabel("dim 1")
                            #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                            plt.legend()
                            # plt.show()
                            # timestr = time.strftime("%Y%m%d-%H%M%S")
                            plt.savefig(os.path.join(path_to_save_plots,'plot_t1_epoch'+str(i)+'_'+str(j)))


                            plt.figure(3, facecolor='w')

                            plt.plot(torch.linspace(0,1,args.n_points),z_p[:,0],c='green', label='model_t0')
                            #plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,0],c='green',s=10)
                            plt.plot(torch.linspace(0,1,args.n_points),z_p[:,1],c='orange', label='model_t1')
                            #plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,1],c='orange',s=10)


                            # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                            plt.scatter(torch.linspace(0,1,obs_test.size(1)),obs_print[:,0],label='Data_t0',c='red', alpha=0.5)
                            plt.scatter(torch.linspace(0,1,obs_test.size(1)),obs_print[:,1],label='Data_t1',c='blue', alpha=0.5)
                            #plt.xlabel("dim 0")
                            #plt.ylabel("dim 1")
                            #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                            plt.legend()
                            # plt.show()
                            # timestr = time.strftime("%Y%m%d-%H%M%S")
                            plt.savefig(os.path.join(path_to_save_plots,'plot_t0t1_epoch'+str(i)+'_'+str(j)))

                            if 'calcium_imaging' in args.experiment_name:
                                # Plot the first 20 frames
                                data_to_plot = obs_print[:20,:]*args.scaling_factor #Get the first 10 samples for a test 
                                predicted_to_plot = z_p[:20,:]*args.scaling_factor
                                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                                predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                                fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                                c=0
                                for idx_row in range (2): 
                                    for idx_col in range(10):
                                        ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row,idx_col].axis('off')
                                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                        ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                        ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+1,idx_col].axis('off')
                                        c+=1
                                fig.tight_layout()
                                plt.savefig(os.path.join(path_to_save_plots, 'plot_first20frame_rec'+str(i)))


                                # Plot the last 20 frames  
                                data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                                predicted_to_plot = z_p[-20:,:]*args.scaling_factor
                                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                                predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                                fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                                c=0
                                for idx_row in range (2): 
                                    for idx_col in range(10):
                                        ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row,idx_col].axis('off')
                                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                        ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                        ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+1,idx_col].axis('off')
                                        c+=1
                                fig.tight_layout()
                                plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_rec'+str(i)))


                                #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                                data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                                predicted_to_plot = z_p[:,:]*args.scaling_factor
                                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                                all_r2_scores = []
                                all_mse_scores = []

                                for idx_frames in range(len(data_to_plot)):
                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                                    all_r2_scores.append(r_value)
                                    # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                                    # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                                    tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                                    all_mse_scores.append(tmp_mse_loss)

                                fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                                ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                                ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                                ax[1].set_xlabel("Frames")
                                ax[0].set_ylabel("R2")
                                ax[1].set_ylabel("MSE")
                                fig.tight_layout()
                                plt.savefig(os.path.join(path_to_save_plots, 'plot_performance_rec'+str(i)))

                                #Plot integral and ode part separated
                                if ode_func is not None and F_func is not None:
                                    Trained_Data_ode = odeint(ode_func,torch.Tensor(obs_print[0,:]).flatten().to(args.device),times.to(args.device),rtol=1e-4,atol=1e-4)
                                    Trained_Data_ode_print = to_np(Trained_Data_ode)
                                    Trained_Data_integral_print  = z_p - Trained_Data_ode_print
                                    # print('Trained_Data_integral_print.max():',np.abs(Trained_Data_integral_print).max())
                                    # print('Trained_Data_ode_print.max():',np.abs(Trained_Data_ode_print).max())

                                    data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                                    predicted_to_plot_ode = Trained_Data_ode_print[-20:,:]*args.scaling_factor
                                    predicted_to_plot_ide = Trained_Data_integral_print[-20:,:]*args.scaling_factor
                                    data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                                    predicted_to_plot_ode = args.fitted_pca.inverse_transform(predicted_to_plot_ode)
                                    predicted_to_plot_ide = args.fitted_pca.inverse_transform(predicted_to_plot_ide)

                                    predicted_to_plot_ode = predicted_to_plot_ode.reshape(predicted_to_plot_ode.shape[0],184, 208) # Add the original frame dimesion as input
                                    predicted_to_plot_ide = predicted_to_plot_ide.reshape(predicted_to_plot_ide.shape[0],184, 208)
                                    data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                                    fig,ax = plt.subplots(6,10, figsize=(15,8), facecolor='w')
                                    c=0
                                    step = 0
                                    for idx_row in range (2): 
                                        for idx_col in range(10):
                                            ax[2*idx_row+step,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                            ax[2*idx_row+step,idx_col].axis('off')

                                            _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ode[c,:].flatten())
                                            ax[2*idx_row+1+step,idx_col].set_title('ODE R2: {:.3f}'.format(r_value**2))
                                            ax[2*idx_row+1+step,idx_col].imshow(predicted_to_plot_ode[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                            ax[2*idx_row+1+step,idx_col].axis('off')

                                            _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ide[c,:].flatten())
                                            ax[2*idx_row+2+step,idx_col].set_title('IDE R2: {:.3f}'.format(r_value**2))
                                            ax[2*idx_row+2+step,idx_col].imshow(predicted_to_plot_ide[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                            ax[2*idx_row+2+step,idx_col].axis('off')
                                            c+=1
                                        step += 1
                                    fig.tight_layout()
                                    plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_ode_vs_ide_rec'+str(i)))

                                    del data_to_plot, predicted_to_plot
                                    del z_to_print, time_to_print, obs_to_print

                            del obs_test, ts_test, z_test, z_p

                            plt.close('all')
                            
                        else:
                            
                            #z_test = z_test.view(args.n_batch,args.n_points,args.n_points,args.time_points,args.dim)
                            if Decoder is not None:
#                                 z_test = z_test.squeeze(-1).permute(0,3,1,2)
#                                 z_test = Decoder(z_test).permute(0,2,3,1)
                                z_test = Decoder(z_test)
                            else:
                                z_test = z_test.view(z_test.shape[0],Data.shape[1],Data.shape[2],args.time_points)
                            if args.initial_t is False:
                                obs_test = obs_test[:,:,:,1:]
                            
                            z_p = z_test
                            if args.n_batch >1:
                                z_p = z_test[0,:,:,:]
                            z_p = to_np(z_p)

                            if args.n_batch >1:
                                obs_print = to_np(obs_test[0,:,:,:])
                            else:
                                obs_print = to_np(obs_test[:,:,:])

                            plot_reconstruction(obs_print, z_p, None, path_to_save_plots, 'plot_epoch_', i, args)
                            
                            plt.close('all')
                            del z_p, z_test, obs_print
#                             plt.figure(1, facecolor='w')
#                             plt.plot(torch.linspace(0,1,args.time_points),z_p[7,5,:],c='green', label='model_t')
                            
                            
#                             plt.scatter(torch.linspace(0,1,obs_test.size(-1)),obs_print[7,5,:],label='Data_t',c='blue', alpha=0.5)
                            
                            
#                             plt.legend()
                            
#                             plt.savefig(os.path.join(path_to_save_plots,'plot_epoch'+str(i)+'_'+str(j)))
                            
#                             del obs_print, z_p
                            
#                             plt.close('all')

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if split_size>0:
                if Encoder is None:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, None, None, None)
                else:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, None, model, Encoder, Decoder)
            else: 
                if Encoder is None:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, model, None, None, None)
                else:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, None, model, Encoder, Decoder)

            #lr_scheduler(loss_validation)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break

        if args.support_tensors is True or args.support_test is True:
                del dummy_times
                
        end = time.time()
        # print(f"Training time: {(end-start)/60:.3f} minutes")
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),Loss_print)
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),Val_Loss)
        # # plt.savefig('trained.png')
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'trained'+timestr))
        # # plt.show()
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'final_losses'+timestr))
        # # plt.show()
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        ## Validating
        model.eval()
        
        t_min , t_max = args.time_interval
        n_points = args.test_points

        
        test_times=torch.sort(torch.rand(n_points),0)[0].to(device)*(t_max-t_min)+t_min
        #test_times=torch.linspace(t_min,t_max,n_points)
        
        #dummy_times = torch.cat([torch.Tensor([0.]).to(device),dummy_times])
        # print('times :',times)
        ###########################################################
        
        with torch.no_grad():
                
            model.eval()
            if Encoder is not None:
                Encoder.eval()
            if Decoder is not None:
                Decoder.eval()
                
            test_loss = 0.0
            loss_list = []
            #counter = 0  

            for j in tqdm(range(0,obs.shape[0],args.n_batch)):
                if args.n_batch==1:
                    if args.eqn_type == 'Burgers':
                        Dataset_all = Dynamics_Dataset(Data[j,:,:],times)
                    else:
                        Dataset_all = Dynamics_Dataset(Data[j,:,:,:],times)
                else:
                    if args.eqn_type == 'Burgers':
                        Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:],times,args.n_batch)
                    else:
                        Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:,:],times,args.n_batch)

                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = args.n_batch)

                obs_test, ts_test, ids_test = Dataset_all.__getitem__(index_np)#next(iter(loader_test))

                ids_test, indices = torch.sort(torch.from_numpy(ids_test))
                # print('indices: ',indices)
                if args.n_batch==1:
                    if args.eqn_type == 'Burgers':
                        obs_test = obs_test[indices,:]
                    else:
                        if Encoder is None:
                            obs_test = obs_test[indices,:,:]
                            obs_test = obs_test[:,indices,:]
                else:
                    if args.eqn_type == 'Burgers':
                        obs_test = obs_test[:,indices,:]
                    else:
                        if Encoder is None:
                            obs_test = obs_test[:,indices,:,:]
                            obs_test = obs_test[:,:,indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                if args.n_batch ==1:
                    if args.eqn_type == 'Burgers':
                        c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:1])
                        interpolation = NaturalCubicSpline(c_coeffs)
                        c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                    else:
                        c = lambda x: obs_test[:,:,:1].repeat(1,1,args.time_points).unsqueeze(-1).to(device)
                else:
                    if args.eqn_type == 'Burgers':
                        c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:,:1])
                        interpolation = NaturalCubicSpline(c_coeffs)
                        c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,1,args.time_points).unsqueeze(-1)
                    else:
#                                 c= lambda x: \
#                                             Encoder(obs_test[:,:,:,:1].repeat(1,1,1,args.time_points)\
#                                             .permute(0,3,1,2)).unsqueeze(-1)\
#                                             .permute(0,2,3,1,4).contiguous().to(args.device)
                        c= lambda x: Encoder(obs_test[:,:,:,:1].permute(0,3,1,2))\
                                    .permute(0,2,3,1).unsqueeze(-2)\
                                    .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)

                if args.eqn_type == 'Navier-Stokes':
#                             y_0 = Encoder(obs_test[:,:,:,:1].repeat(1,1,1,args.time_points)\
#                             .permute(0,3,1,2)).unsqueeze(-1)\
#                             .permute(0,2,3,1,4)[:,:,:,:1,:]
                    y_0 = Encoder(obs_test[:,:,:,:1].permute(0,3,1,2))\
                                    .permute(0,2,3,1).unsqueeze(-2)\
                                    .to(args.device)

                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)

                if args.support_test is False:
                    if args.n_batch==1:
                        if args.eqn_type == 'Burgers':
                            z_test = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_test[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                        else:
                            z_test = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_test[:,:,0].unsqueeze(-1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()

                    else:
                        if args.eqn_type == 'Burgers':
                            z_test = Integral_spatial_attention_solver_multbatch(
                                torch.linspace(0,1,args.time_points).to(device),
                                obs_test[:,0].unsqueeze(-1).to(args.device),
                                c=c,
                                sampling_points = args.time_points,
                                mask=mask,
                                Encoder = model,
                                max_iterations = args.max_iterations,
                                spatial_integration=True,
                                spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                spatial_domain_dim=1,
                                #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                smoothing_factor=args.smoothing_factor,
                                use_support=False,
                                ).solve()
                        else:
                            z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(args.device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                else:
                    z_test = Integral_spatial_attention_solver(
                            torch.linspace(0,1,args.time_points).to(device),
                            obs_[0].unsqueeze(1).to(args.device),
                            c=c,
                            sampling_points = args.time_points,
                            support_tensors=dummy_times.to(device),
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            spatial_integration=True,
                            spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                            spatial_domain_dim=1,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            smoothing_factor=args.smoothing_factor,
                            output_support_tensors=True,
                            ).solve()




                if Decoder is not None:
                    z_test = Decoder(z_test)
                else:
                    z_test = z_test.view(args.n_batch,Data.shape[1],Data.shape[2],args.time_points)
                if args.initial_t is False:
                    obs_test = obs_test[...,1:]
                    z_test = z_test[...,1:]
                    
                mse_error = F.mse_loss(z_test, obs_test.detach())
                
                test_loss += mse_error.item()
                loss_list.append(mse_error.item())
                
#                 for in_batch_indx in range(args.n_batch):

#                     obs_print = to_np(obs_test[in_batch_indx,:,:,:])
#                     z_p = to_np(z_test[in_batch_indx,:,:,:])

#                     #plot_reconstruction(obs_print, z_p, None, path_to_save_plots, 'plot_epoch_', i, args)
#                     plot_reconstruction(obs_print, z_p, None, None, None, None, args)

#                     plt.close('all')
#                     del z_p, obs_print
                del z_test, obs_test
            
            print(loss_list)
            print("Average loss: ",test_loss*args.n_batch/obs.shape[0])
            

                        
    