# Requires KUBECONFIG in the environment: export KUBECONFIG=...
NAMESPACE  ?= kimi-k3

.PHONY: apply status pods logs

apply:
	kubectl apply -f manifests/

status:
	kubectl get all,pvc,cm,secret -n $(NAMESPACE)

pods:
	kubectl get pods -n $(NAMESPACE) -o wide

logs:
	kubectl logs -n $(NAMESPACE) -l app.kubernetes.io/name=kimi-k3 --tail=100 -f
