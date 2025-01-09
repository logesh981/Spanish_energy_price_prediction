terraform {
  required_providers {
    aws = {
        source = "hashicorp/aws"
        version = "~> 5.0"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = "~> 1.14.0"
    }
  }
}
provider "aws" {
  region = "us-east-1"
}
provider "kubectl" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.cluster.token
  load_config_file       = false
}
