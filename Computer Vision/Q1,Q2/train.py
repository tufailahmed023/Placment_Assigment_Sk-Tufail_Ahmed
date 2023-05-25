def training(model,epoch,train_path,val_path,tain_loder,val_loder):
	from torch.optim import Adam
	import torch.nn as nn

	model = model
	num_epochs = epoch 

	optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
	loss_function=nn.CrossEntropyLoss()


	train_count=len(glob.glob(train_path+'/**/*.jpg'))
	val_count=len(glob.glob(val_path+'/**/*.jpg'))

	best_accuracy=0.0

	for epoch in range(num_epochs):
		
		#Evaluation and training on training dataset
		model.train()
		train_accuracy=0.0
		train_loss=0.0
		
		for i, (images,labels) in enumerate(train_loader):
			if torch.cuda.is_available():
				images=Variable(images.cuda())
				labels=Variable(labels.cuda())
				
			optimizer.zero_grad()
			
			outputs=model(images)
			loss=loss_function(outputs,labels)
			loss.backward()
			optimizer.step()
			
			
			train_loss+= loss.cpu().data*images.size(0)
			_,prediction=torch.max(outputs.data,1)
			
			train_accuracy+=int(torch.sum(prediction==labels.data))
			
		train_accuracy=train_accuracy/train_count
		train_loss=train_loss/train_count
		
		
		# Evaluation on testing dataset
		model.eval()
		
		val_accuracy=0.0
		for i, (images,labels) in enumerate(val_loader):
			if torch.cuda.is_available():
				images=Variable(images.cuda())
				labels=Variable(labels.cuda())
				
			outputs=model(images)
			_,prediction=torch.max(outputs.data,1)
			val_accuracy+=int(torch.sum(prediction==labels.data))
		
		val_accuracy=val_accuracy/val_count
		
		
		print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' VAl Accuracy: '+str(val_accuracy))
		
		#Save the best model
		if val_accuracy>best_accuracy:
			torch.save(model.state_dict(),'best_checkpoint.model')
			best_accuracy=val_accuracy
