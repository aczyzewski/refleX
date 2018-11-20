// Select dropdowns
if ($('select').length) {
    // Traverse through all dropdowns
    $.each($('select'), function (i, val) {
        var $el = $(val);

        // If there's any dropdown with default option selected
        // give them `not_chosen` class to style appropriately
        // We assume default option to have a value of '' (empty string)
        if (!$el.val()) {
            $el.addClass("not_chosen");
        }

        // Add change event handler to do the same thing,
        // i.e., adding/removing classes for proper
        // styling. Basically we are emulating placeholder
        // behaviour on select dropdowns.
        $el.on("change", function () {
            if (!$el.val())
                $el.addClass("not_chosen");
            else
                $el.removeClass("not_chosen");
        });

        // end of each callback
  });
}
