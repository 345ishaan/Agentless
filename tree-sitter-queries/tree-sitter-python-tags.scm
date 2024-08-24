(class_definition
  name: (identifier) @name.definition.class) @definition.class

(function_definition
  name: (identifier) @name.definition.function) @definition.function

(call
  function: [
      (identifier) @name.reference.call
      (attribute
        attribute: (identifier) @name.reference.call)
  ]) @reference.call

(import_statement
  name: (dotted_name) @name.import) @import

(import_from_statement
  module_name: (dotted_name) @name.import.from) @import.from

(import_from_statement
  name: (dotted_name (identifier) @name.import)) @import.from