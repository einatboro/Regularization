library(MASS)
library(leaps)
library(broom)
library(glmnet)

vanilla_df = read.csv("diabetes.txt", header = TRUE,sep = "\t")

df <- as.data.frame(scale(vanilla_df, center = TRUE, scale = TRUE))

head(df)

cols <- colnames(df[,-11])
################################OLS model############################################################
frmla <- as.formula(paste("Y~",(paste(cols,collapse = "+"))))
ml <- NULL
ml_list <- vector(mode="list", length=9)
ml_list[[1]] <- lm(frmla,df) 
AIC(ml_list[[1]])

################################stepwise AIC minimization model############################################################

ml_list[[2]] <- stepAIC(ml_list[[1]]) 
AIC(ml_list[[2]])


################################exhaustive Cp minimization model###########################################################
ml_list[[3]] <- leaps( x=df[,-11], y=df[,11], names=cols, method="Cp") 
cbind(ml_list[[3]]$size,ml_list[[3]]$which, ml_list[[3]]$Cp)
best.model.Cp = ml_list[[3]]$which[which((ml_list[[3]]$Cp == min(ml_list[[3]]$Cp))),]
best.model.Cp


################################ridge model###########################################################
grid = 10^seq(10, -2, length = 100)
ml_list[[4]] <- cv.glmnet(as.matrix(df[,-11]), df[,11], alpha = 0, lambda= grid,intercept=FALSE) 
ml_list[[4]]$lambda.min 
ml_list[[4]]$lambda.1se
indMin_r <- which.min(ml_list[[4]]$cvm)
plot(ml_list[[4]])
abline(h = ml_list[[4]]$cvm[indMin_r] + c(0, ml_list[[4]]$cvsd[indMin_r]))

(coef(ml_list[[4]], s = ml_list[[4]]$lambda.min))[-1]
(coef(ml_list[[4]], s = ml_list[[4]]$lambda.1se))[-1]
mse.min_ridge <- ml_list[[4]]$cvm[ml_list[[4]]$lambda == ml_list[[4]]$lambda.min]
mse.1se_ridge <- ml_list[[4]]$cvm[ml_list[[4]]$lambda == ml_list[[4]]$lambda.1se]

################################elastic net model############################################################
a <- 0.5 
search_min <- foreach(i = a, .combine = rbind) %dopar% {
  cv <- cv.glmnet(as.matrix(df[,-11]), df[,11], alpha = i)
  data.frame(cvm = cv$cvm[cv$lambda == cv$lambda.min], lambda.min = cv$lambda.min, alpha = i)
}
cv <- NULL
search_1se <- foreach(i = a, .combine = rbind) %dopar% {
  cv <- cv.glmnet(as.matrix(df[,-11]), df[,11], alpha = i)
  data.frame(cvm = cv$cvm[cv$lambda == cv$lambda.1se], lambda.1se = cv$lambda.1se, alpha = i)
}
cv_min <- search_min[search_min$cvm == min(search_min$cvm), ]
cv_1se <- search_1se[search_1se$cvm == min(search_1se$cvm), ]
indMin_en_min <- which.min(search_min$cvm)
indMin_en_cv1 <- which.min(search_1se$cvm)
ml_list[[5]] <- cv.glmnet(as.matrix(df[,-11]), df[,11], alpha = search_min[indMin_en_min,3])
ml_list[[6]] <- cv.glmnet(as.matrix(df[,-11]), df[,11], alpha = search_min[indMin_en_cv1,3])


(coef(ml_list[[5]], s = ml_list[[5]]$lambda.min))[-1]
(coef(ml_list[[6]], s = ml_list[[6]]$lambda.1se))[-1]

mse.min_elastic <- ml_list[[5]]$cvm[ml_list[[5]]$lambda == ml_list[[5]]$lambda.min]
mse.1se_elastic <- ml_list[[6]]$cvm[ml_list[[6]]$lambda == ml_list[[6]]$lambda.1se]

plot(ml_list[[5]])
abline(h = ml_list[[5]]$cvm[indMin_en_min] + c(0, ml_list[[5]]$cvsd[indMin_en_min]))
plot(ml_list[[6]])
abline(h = ml_list[[6]]$cvm[indMin_en_cv1] + c(0, ml_list[[6]]$cvsd[indMin_en_cv1]))

################################lasso model############################################################
ml_list[[7]] <- cv.glmnet(as.matrix(df[,-11]), df[,11], alpha = 1, lambda= grid,intercept=FALSE) 
ml_list[[7]]$lambda.min 
ml_list[[7]]$lambda.1se
indMin_l <- which.min(ml_list[[7]]$cvm)
(coef(ml_list[[7]], s = ml_list[[7]]$lambda.min))[-1]
(coef(ml_list[[7]], s = ml_list[[7]]$lambda.1se))[-1]
plot(ml_list[[7]])
abline(h = ml_list[[7]]$cvm[indMin_l] + c(0, ml_list[[7]]$cvsd[indMin_l]))
mse.min_lasso <- ml_list[[7]]$cvm[ml_list[[7]]$lambda == ml_list[[7]]$lambda.min]
mse.1se_lasso <- ml_list[[7]]$cvm[ml_list[[7]]$lambda == ml_list[[7]]$lambda.1se]

################################relaxed lasso model############################################################

fit_min <- glmnet(as.matrix(df[,-11]), df[,11], lambda = ml_list[[7]]$lambda.min )
fit_1se <- glmnet(as.matrix(df[,-11]), df[,11], lambda = ml_list[[7]]$lambda.1se )
active.variables_min <- sapply(coef(fit_min)[-1,], function(x){which(x != 0)})
active.variables_1se <- sapply(coef(fit_1se)[-1,], function(x){which(x != 0)})
av_min <- names(active.variables_min [active.variables_min ==1])[!is.na(names(active.variables_min [active.variables_min ==1]))]
av_1se <- names(active.variables_1se [active.variables_1se ==1])[!is.na(names(active.variables_1se [active.variables_1se ==1]))]
frmla_min <- as.formula(paste("Y~",(paste(av_min,collapse = "+"))))
frmla_1se <- as.formula(paste("Y~",(paste(av_1se,collapse = "+"))))
ml_list[[8]] <- lm(frmla_min,df)
ml_list[[9]] <- lm(frmla_1se,df)
AIC(ml_list[[8]])
AIC(ml_list[[9]])

x <- df[,-11]
y <- df[,11]
n <- dim(x)[1]
p <- dim(x)[2]


# Soft-thresholding function
soft <- function(K, thr) {
  if (K < - thr) {
    soft <- K + thr
  }
  else if (K >  thr)
  {
    soft <- K - thr
  }
  else
  {
    soft <- 0
  }
}

# Elastic net via cyclic coordinate descent
# x and y should be centered and scaled!
en <- function(x, y, lambda, alpha, tol=1e-7) {
  beta <- ml_list[[1]]$coefficients[-1]
  ssdiff <- 2*tol
  while (ssdiff>tol) {  # outer loop
    beta.old <- beta
    
    # Write an inner loop to update each component of beta:
    # i.e., for j=1,...,ncol(x), update beta_j to expression (3) in the assignment.
    # Call the above function soft() for soft thresholding.
    for(j in 1:p){
      curr_x <- x[,-j]     
      curr_beta <- beta[-j]  
      
      curr_r <- y - as.matrix(curr_x) %*% curr_beta
      
      soft_arg <- sum(x[,j]*curr_r)
      beta[j] <- soft(soft_arg, n*lambda*alpha) / (sum( x[,j] ^ 2) +n*lambda*(1-alpha))
    }
    
    
    
    ssdiff <- sum((beta.old-beta)^2) # sum of squared changes to the coefficients
    # when this is < tol, outer loop terminates		   
  }
  return(beta)
}



objective <- function(x, y, lambda, alpha, beta) {
  # Compute objective function in equation (1) of the assignment.
  sum((y-as.matrix(x)%*%beta)^2) + lambda * 0.5*(1 - alpha)*sum(beta^2)+ alpha*sum(abs(beta))
}


small.lambda.index_min <- which(ml_list[[5]]$lambda == ml_list[[5]]$lambda.min)
small.lambda.betas_min <- ml_list[[5]]$glmnet.fit$beta[, small.lambda.index_min]
calc_beta_min <- en(x=x,y=y,lambda=ml_list[[5]]$lambda.min,alpha=0.5,tol = 10^-7)
cbind(small.lambda.betas_min,calc_beta_min)
sum(small.lambda.betas_min-calc_beta_min)

small.lambda.index_1se <- which(ml_list[[6]]$lambda == ml_list[[6]]$lambda.1se)
small.lambda.betas_1se <- ml_list[[6]]$glmnet.fit$beta[, small.lambda.index_1se]
calc_beta_1se <- en(x=x,y=y,lambda=ml_list[[6]]$lambda.1se,alpha=0.5,tol = 10^-7)
cbind(small.lambda.betas_1se,calc_beta_1se)
sum(small.lambda.betas_1se-calc_beta_1se)

objective(x=x,y=y,lambda=ml_list[[5]]$lambda.min,alpha=0.5,beta=calc_beta_min)
objective(x=x,y=y,lambda=ml_list[[5]]$lambda.min,alpha=0.5,beta=small.lambda.betas_min)

objective(x=x,y=y,lambda=ml_list[[6]]$lambda.1se,alpha=0.5,beta=calc_beta_1se)
objective(x=x,y=y,lambda=ml_list[[6]]$lambda.1se,alpha=0.5,beta=small.lambda.betas_1se)

