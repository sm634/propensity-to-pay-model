# ml_model_cloud_run: Name of the resource in Terraform
# ml-model-cloud-run: Stored with this name in Container Registry
# google_cloud_run_service: type of the resource

resource "google_cloud_run_service" "cloud_run_ml" {
  name     = "cloud-run-ml"
  project  = var.project_id
  location = var.region

  template {
    spec {
      containers {
        image = var.image_uri
      }
    service_account_name = var.service_account_name
    }
  }

  # traffic {
  #   percent         = 100
  #   latest_revision = true
  # }
}