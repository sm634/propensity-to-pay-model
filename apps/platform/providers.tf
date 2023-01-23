# Terraform plugins called providers let Terraform interact (to manager the infrastructure) with cloud platforms and other services via their application programming interfaces (APIs)

terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "4.36.0"
    }
  }
  backend "gcs" {
    bucket = "vertex-ai-test-admin"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}