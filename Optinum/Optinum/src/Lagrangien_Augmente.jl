@doc doc"""
Résolution des problèmes de minimisation sous contraintes d'égalités

# Syntaxe
```julia
Lagrangien_Augmente(algo,fonc,contrainte,gradfonc,hessfonc,grad_contrainte,
			hess_contrainte,x0,options)
```

# Entrées
  * **algo** 		   : (String) l'algorithme sans contraintes à utiliser:
    - **"newton"**  : pour l'algorithme de Newton
    - **"cauchy"**  : pour le pas de Cauchy
    - **"gct"**     : pour le gradient conjugué tronqué
  * **fonc** 		   : (Function) la fonction à minimiser
  * **contrainte**	   : (Function) la contrainte [x est dans le domaine des contraintes ssi ``c(x)=0``]
  * **gradfonc**       : (Function) le gradient de la fonction
  * **hessfonc** 	   : (Function) la hessienne de la fonction
  * **grad_contrainte** : (Function) le gradient de la contrainte
  * **hess_contrainte** : (Function) la hessienne de la contrainte
  * **x0** 			   : (Array{Float,1}) la première composante du point de départ du Lagrangien
  * **options**		   : (Array{Float,1})
    1. **epsilon** 	   : utilisé dans les critères d'arrêt
    2. **tol**         : la tolérance utilisée dans les critères d'arrêt
    3. **itermax** 	   : nombre maximal d'itération dans la boucle principale
    4. **lambda0**	   : la deuxième composante du point de départ du Lagrangien
    5. **mu0,tho** 	   : valeurs initiales des variables de l'algorithme

# Sorties
* **xmin**		   : (Array{Float,1}) une approximation de la solution du problème avec contraintes
* **fxmin** 	   : (Float) ``f(x_{min})``
* **flag**		   : (Integer) indicateur du déroulement de l'algorithme
   - **0**    : convergence
   - **1**    : nombre maximal d'itération atteint
   - **(-1)** : une erreur s'est produite
* **niters** 	   : (Integer) nombre d'itérations réalisées

# Exemple d'appel
```julia
using LinearAlgebra
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
algo = "gct" # ou newton|gct
x0 = [1; 0]
options = []
contrainte(x) =  (x[1]^2) + (x[2]^2) -1.5
grad_contrainte(x) = [2*x[1] ;2*x[2]]
hess_contrainte(x) = [2 0;0 2]
output = Lagrangien_Augmente(algo,f,contrainte,gradf,hessf,grad_contrainte,hess_contrainte,x0,options)
```
"""
function Lagrangien_Augmente(algo,fonc::Function,contrainte::Function,gradfonc::Function,
	hessfonc::Function,grad_contrainte::Function,hess_contrainte::Function,x0,options)

  if options == []
		epsilon = 1e-8	
		itermax = 1000
		lambda0 = 2
		mu0 = 100
    tho = 2
    Tol_abs = sqrt(eps())
    Tol_rel = 1e-15
    alpha = 0.1
    beta = 0.9
    nu0_chapeau = 0.1258925
    eps_0 = 1/mu0
    else
epsilon = options[1]
		tol = options[2]
		itermax = options[3]
		lambda0 = options[4]
		mu0 = options[5]
    tho = options[6]
    Tol_abs = options[7]
    Tol_rel = options[8]
    alpha = options[9]
    beta = options[10]
    nu0_chapeau = options[11]
    eps_0 = options[12]
  end

  nb_iters = 0

  xk = x0
  lambda_k = lambda0
  mu_k = mu0
  nu_k = nu0_chapeau/(mu0^alpha)
  eps_k = eps_0
  
  while(nb_iters < itermax)
  
    function LA(x)
      return fonc(x) + transpose(lambda_k)*contrainte(x)+(mu_k/2)*(norm(contrainte(x)))^2
    end
  
    function grad_LA(x)
      return (gradfonc(x) + grad_contrainte(x)*lambda_k + mu_k*grad_contrainte(x)*contrainte(x))
    end
  
    function hess_LA(x)
      return hessfonc(x) + transpose(lambda_k)*hess_contrainte(x) + mu_k*hess_contrainte(x)*contrainte(x) +
      mu_k*grad_contrainte(x)*transpose(grad_contrainte(x))
    end
    
    # [100, eps_k, 0]
            if algo == "newton"
                xk1, ~ = Algorithme_De_Newton(LA,grad_LA,hess_LA,xk,[])
            elseif algo == "cauchy"
                xk1 , ~ = Regions_De_Confiance("cauchy",LA,grad_LA,hess_LA,xk,[])
            elseif algo=="gct"
                xk1, ~ = Regions_De_Confiance("gct",LA,grad_LA,hess_LA,xk,[])
            else
                flag = -1
            end


    if (norm(grad_LA(xk1))<= max(Tol_rel*norm(grad_LA(x0)),Tol_abs) && 
    norm(contrainte(xk))<=max(Tol_rel*contrainte(x0), Tol_abs) )
      break
    else
      if (norm(contrainte(xk1)) <= nu_k)
        lambda_k1 = lambda_k + mu_k*contrainte(xk1)
        mu_k1 = mu_k
        eps_k1 = eps_k/mu_k
        nu_k1 = nu_k/(mu_k^beta)

      else
        lambda_k1 = lambda_k
        mu_k1 = tho*mu_k
        eps_k1 = eps_0/mu_k1
        println(nu0_chapeau/((mu_k1)^alpha))
      end

      lambda_k = lambda_k1
      mu_k = mu_k1
      nu_k = nu_k1
      xk = xk1

      nb_iters = nb_iters+1
    end
  end  


   n = length(x0)
   xmin = xk1
   fxmin = fonc(xmin)
   flag = 0
   if (nb_iters >= itermax)
        flag = 1
   end

    return xmin,fxmin,flag,nb_iters
end

