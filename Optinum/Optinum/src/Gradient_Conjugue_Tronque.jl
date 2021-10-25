@doc doc"""
Minimise le problème : ``min_{||s||< \delta_{k}} q_k(s) = s^{t}g + (1/2)s^{t}Hs``
                        pour la ``k^{ème}`` itération de l'algorithme des régions de confiance

# Syntaxe
```julia
sk = Gradient_Conjugue_Tronque(fk,gradfk,hessfk,option)
```

# Entrées :   
   * **gradfk**           : (Array{Float,1}) le gradient de la fonction f appliqué au point xk
   * **hessfk**           : (Array{Float,2}) la Hessienne de la fonction f appliqué au point xk
   * **options**          : (Array{Float,1})
      - **delta**    : le rayon de la région de confiance
      - **max_iter** : le nombre maximal d'iterations
      - **tol**      : la tolérance pour la condition d'arrêt sur le gradient


# Sorties:
   * **s** : (Array{Float,1}) le pas s qui approche la solution du problème : ``min_{||s||< \delta_{k}} q(s)``

# Exemple d'appel:
```julia
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
xk = [1; 0]
options = []
s = Gradient_Conjugue_Tronque(gradf(xk),hessf(xk),options)
```
"""
function Gradient_Conjugue_Tronque(gradfk,hessfk,options)

    "# Si option est vide on initialise les 3 paramètres par défaut"
    if options == []
        deltak = 2
        max_iter = 100
        tol = 1e-6
    else
        deltak = options[1]
        max_iter = options[2]
        tol = options[3]
    end
    n = length(gradfk)
    if(n==1)
        s=0
        sj=0
    else
        s = zeros(n)
        sj = zeros(n)
    end
    gj = gradfk
    pj = -gradfk
    nb_iters = 0
    while(nb_iters < max_iter) 
        kj = transpose(pj) * hessfk * pj
        if( kj <= 0)
            # norm(sj)^2+2*pj*sj*x+(norm(sj)^2-delta^2) = 0
            # Résultat trouvé est 
            a = norm(pj)^2
            b = 2*transpose(sj)*pj
            c = norm(sj)^2-norm(deltak)^2
            delta_sol = sqrt(b^2-4*a*c)
            sol1 = (-b + delta_sol)/(2*a)
            sol2 = (-b - delta_sol)/(2*a)
            # puisqu'on doit minimiser la fct q_k(sj + sol*pj) = s^{t}g + (1/2)s^{t}Hs
            q_sol1 = transpose(sj + sol1*pj)*gj + (1/2)* transpose(sj + sol1*pj)* hessfk *(sj + sol1*pj)
            q_sol2 = transpose(sj + sol2*pj)*gj + (1/2)* transpose(sj + sol2*pj)* hessfk *(sj + sol2*pj)
            if(q_sol1<q_sol2)
                sigma = sol1
            else
                sigma = sol2
            end
            s = sj + sigma*pj
            break
        end
        
        alphaj = (transpose(gj)*gj)/kj
        if ( norm(sj + alphaj*pj) >= deltak)
             # norm(sj)^2+2*pj*sj*x+(norm(sj)^2-delta^2) = 0
            # Résultat trouvé est 
            a = norm(pj)^2
            b = 2*transpose(sj)*pj
            c = norm(sj)^2-norm(deltak)^2
            delta_sol = sqrt(b^2-4*a*c)
            sol1 = (-b + delta_sol)/(2*a)
            sol2 = (-b - delta_sol)/(2*a)
            # la racine positive 
            if(sol1>=0)
                sigma = sol1
            else
                sigma = sol2
            end
            s =sj + sigma*pj
            break
        end
        sj1 = sj + alphaj*pj
        gj1 = gj + alphaj* hessfk*pj
        betaj = (transpose(gj1)*gj1)/(transpose(gj)*gj)
        pj1 = -gj1 + betaj*pj
        sj=sj1
        gj=gj1
        pj=pj1
        if(norm(gj) < tol*norm(gradfk))
            s = sj1
            break
        end
        nb_iters = nb_iters+1
    end
   return s
end
