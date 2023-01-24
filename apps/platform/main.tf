resource "google_notebooks_instance" "instance" {
  name = "test1-propensity-model-train-tf"
  location = "europe-west2-a"
  machine_type = "n1-standard-4"
  vm_image {
    project      =  var.project_id
    image_family = "tf-ent-latest-cpu"
  }
}