
"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("estimation-form");
  const brandSelect = document.getElementById("make_norm");
  const modelSelect = document.getElementById("model_norm");
  const modelsElement = document.getElementById("models-data");
  const demoElement = document.getElementById("demo-data");
  const demoButton = document.getElementById("load-demo");
  const errorBlock = document.getElementById("form-error");

  if (errorBlock) {
    errorBlock.focus();
  }

  if (!brandSelect || !modelSelect || !modelsElement) {
    return;
  }

  let modelsByBrand = {};
  let demoData = {};

  try {
    modelsByBrand = JSON.parse(
      modelsElement.textContent
    );
  } catch (error) {
    console.error(
      "Impossible de charger les modèles.",
      error
    );
    return;
  }

  if (demoElement) {
    try {
      demoData = JSON.parse(
        demoElement.textContent
      );
    } catch (error) {
      console.error(
        "Impossible de charger l'exemple.",
        error
      );
    }
  }

  function updateModels(selectedModel = "") {
    const brand = brandSelect.value;
    const models = modelsByBrand[brand] || [];

    modelSelect.innerHTML = "";

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = models.length
      ? "Sélectionner un modèle"
      : "Aucun modèle disponible";
    modelSelect.appendChild(placeholder);

    models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      option.selected = model === selectedModel;
      modelSelect.appendChild(option);
    });

    modelSelect.disabled = models.length === 0;
  }

  function setFieldValue(name, value) {
    if (!form || value === null || value === undefined) {
      return;
    }

    const field = form.elements.namedItem(name);

    if (!field) {
      return;
    }

    field.value = String(value);
    field.dispatchEvent(
      new Event("change", { bubbles: true })
    );
  }

  brandSelect.addEventListener("change", () => {
    updateModels("");
  });

  const initiallySelectedModel =
    modelSelect.dataset.selected || "";

  updateModels(initiallySelectedModel);

  if (demoButton) {
    demoButton.addEventListener("click", () => {
      setFieldValue(
        "make_norm",
        demoData.make_norm
      );

      updateModels(
        demoData.model_norm || ""
      );

      Object.entries(demoData).forEach(
        ([name, value]) => {
          if (
            name !== "make_norm"
            && name !== "model_norm"
          ) {
            setFieldValue(name, value);
          }
        }
      );

      modelSelect.value = (
        demoData.model_norm || ""
      );

      form.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });

      const firstField = form.querySelector(
        "select, input"
      );

      if (firstField) {
        firstField.focus();
      }
    });
  }
});
